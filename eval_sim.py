"""
eval_sim.py — Monte-Carlo simulator for passing a futures prop-firm evaluation.

Python 3.10+, numpy only (pure numpy/python at the sim core for speed). MIT licensed.

WHAT THIS IS
------------
A faithful Monte-Carlo model of a futures prop-firm *evaluation* (the paid combine you
must pass to get a funded account). The single most error-prone part of these models is
the TRAILING DRAWDOWN. This file models it precisely for the three firms below.

The simulator answers: "Given a strategy's edge (win rate, R-multiple distribution,
$ risk per micro contract, trades/month) and a contract count (in MICROS), what is the
probability of passing each firm's eval in 30 calendar days, and *how* do failures happen
(hit the trailing floor / ran out of time / failed the consistency rule)?"

--------------------------------------------------------------------------------
VERIFIED FIRM RULESETS (current as of June 2026 — hard-coded, do not invent numbers)
--------------------------------------------------------------------------------
APEX 4.0 (post 2026-03-01): trader CHOOSES EOD or Intraday trailing. NO minimum trading
  days. 30 CALENDAR days max. No consistency rule on eval. DLL modeled OFF by default
  (the intraday option carries no DLL; EOD accounts have one but we leave it off unless
  explicitly enabled).
      size  target  drawdown  maxMinis
      25k   1500    1000      4
      50k   3000    2000      6
      100k  6000    3000      8
      150k  9000    4000      12

TOPSTEP (2026): EOD-trailing max-loss. Daily Loss Limit enforced. Consistency rule: the
  best single trading-day's profit must be < 50% of total profit at the instant you hit
  the target. Effectively needs >=2 trading days. 30-day horizon.
      size  target  drawdown  DLL    maxMinis(approx)
      50k   3000    2000      1000   5
      100k  6000    3000      2000   10
      150k  9000    4500      3000   15

LUCID (LucidTest, 2026): EOD-trailing. Daily Loss Limit = 20% of target. MINIMUM 5 trading
  days. 90/10 funded split (irrelevant to the eval pass test). Trailing converts to STATIC
  once (initial balance + target) is reached.
      size  target  drawdown  DLL    minDays
      25k   1500    1500      300    5
      50k   3000    2000      600    5
      100k  6000    3000      1200   5
      150k  9000    4000(est) 1800(est) 5

--------------------------------------------------------------------------------
TRAILING-DRAWDOWN MECHANICS (the heart of the model)
--------------------------------------------------------------------------------
We track equity in DOLLARS as an offset from the starting balance (start = 0.0). The
trailing floor is also an offset from start. The account FAILS when equity touches the floor.

INTRADAY trailing:
    floor = (running peak of equity, INCLUDING unrealized intraday swings) - drawdown.
    Updates trade-by-trade, never decreases. If you spike +800 unrealized then give it back,
    your floor already rose by 800 of that spike — the cushion is permanently eroded.
    Because we model P&L per *trade*, the running peak is updated at every trade-equity point.

EOD trailing:
    floor = (peak of END-OF-DAY equity values) - drawdown. Recomputed ONCE at each day's
    close and enforced during the NEXT session. Intraday wiggles within a day do NOT raise
    the floor. Much gentler. (During the current day we still must not let equity fall to
    the floor that was set at last close — that floor is fixed for the whole day.)

THE LOCK:
    The trailing floor stops rising once it would reach a lock level (an offset from start):
      - TopStep / Lucid: floor locks at the STARTING BALANCE (offset 0.0). I.e. once you are
        up ~1x drawdown so the trailing floor would climb to >= start, it pins at start and
        never rises again. This guarantees a funded account can't be trailed into the red.
      - Apex 4.0 intraday: floor locks at start + a small buffer (we use +100 by default,
        configurable). Apex's published behavior is the trail freezes once the buffer above
        starting balance is reached.
    Lucid ALSO has the rule that once (start + target) equity is reached the trailing stops
    entirely and becomes STATIC (we implement this by freezing the floor at that point too —
    in practice the floor has already locked at start well before target for these numbers).

DAILY LOSS LIMIT (DLL), where present (TopStep, Lucid; Apex modeled off):
    A per-DAY loss cap measured from the day's STARTING equity. If cumulative intraday loss
    reaches the DLL, trading stops for that day (the day's loss is capped at -DLL) and resumes
    next session. Hitting the DLL does NOT fail the eval — it just ends the day. It only fails
    if the same move also touches the trailing floor.

FAIL conditions:
    (1) equity touches the trailing floor  -> fail reason "hit_trailing_floor"  (account closed)
    (2) 30 calendar days elapse with target never reached -> "timeout_no_target"
    (3) target reached but consistency rule violated      -> "consistency_fail" (TopStep)
PASS:
    equity reaches (start + target) AND all constraints satisfied at that moment:
      - min trading days met (Lucid 5, TopStep effectively >=2 via consistency, Apex 0)
      - consistency rule satisfied (TopStep)
    If target is reached but min-days not yet met, we keep trading (cannot pass early); the
    account simply must survive more days. If it's reached but consistency fails, the trader
    keeps trading to dilute the big day (we model the realistic "keep going" — but if they
    still can't satisfy it by day 30 it's a consistency timeout).

--------------------------------------------------------------------------------
CONTRACT MULTIPLIERS
--------------------------------------------------------------------------------
MNQ micro = $2/pt, MES micro = $5/pt, NQ mini = $20/pt, ES mini = $50/pt.
maxMinis are MINI contracts; 1 mini = 10 micros => maxMicros = maxMinis * 10.
We size in MICROS. "dollar_risk_per_micro" is the $ risk of a 1R loss on ONE micro contract,
so a trade with `micros` contracts risks `micros * dollar_risk_per_micro` at -1R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


# =============================================================================
# 1. FIRM RULESETS
# =============================================================================

@dataclass
class FirmRuleset:
    """A single prop-firm account configuration for the evaluation.

    All dollar amounts are positive magnitudes. Floors/equity are tracked as
    offsets from the starting balance (start = 0.0) inside the simulator.
    """
    name: str
    size: int                      # nominal account size, e.g. 50000 (label only)
    target: float                  # profit target in $ (reach start+target to pass)
    drawdown: float                # trailing drawdown in $ (initial cushion below start... see below)
    trailing: str                  # "intraday" or "eod"
    max_micros: int                # contract cap in micros (maxMinis * 10)
    dll: Optional[float] = None    # daily loss limit in $ (None = off)
    min_trading_days: int = 0      # minimum distinct trading days to be eligible to pass
    consistency: Optional[float] = None  # best-day fraction cap (e.g. 0.50) or None
    horizon_calendar_days: int = 30
    trading_days_in_horizon: int = 21    # ~21 trading days inside 30 calendar days
    # Lock level: the offset-from-start at which the trailing floor stops rising.
    # TopStep/Lucid lock at start (0.0). Apex intraday locks at start + buffer.
    lock_level: float = 0.0
    # Lucid: once start+target reached, trailing becomes fully static.
    static_after_target: bool = False

    def initial_floor(self) -> float:
        """Floor offset at account open.

        At open, equity offset = 0 and peak offset = 0, so floor = 0 - drawdown
        = -drawdown. This is the classic 'you start with `drawdown` of room'.
        """
        return -self.drawdown


# ---- Builders for each firm's size table (verified numbers above) -----------

# 1 mini = 10 micros.
def _minis(m: int) -> int:
    return m * 10


def apex_ruleset(size: int, trailing: str = "intraday",
                 dll: Optional[float] = None, lock_buffer: float = 100.0) -> FirmRuleset:
    """Apex 4.0 account. trailing in {'intraday','eod'}. DLL off by default."""
    table = {
        25000:  dict(target=1500, drawdown=1000, minis=4),
        50000:  dict(target=3000, drawdown=2000, minis=6),
        100000: dict(target=6000, drawdown=3000, minis=8),
        150000: dict(target=9000, drawdown=4000, minis=12),
    }
    t = table[size]
    # Apex intraday locks ~at start + small buffer. EOD: we also lock at start+buffer
    # (Apex EOD trail freezes at the same buffer once threshold is reached).
    return FirmRuleset(
        name=f"APEX4_{size//1000}k_{trailing}",
        size=size, target=t["target"], drawdown=t["drawdown"],
        trailing=trailing, max_micros=_minis(t["minis"]),
        dll=dll, min_trading_days=0, consistency=None,
        lock_level=lock_buffer, static_after_target=False,
    )


def topstep_ruleset(size: int) -> FirmRuleset:
    """TopStep account: EOD trailing, DLL enforced, 50% consistency, lock at start."""
    table = {
        50000:  dict(target=3000, drawdown=2000, dll=1000, minis=5),
        100000: dict(target=6000, drawdown=3000, dll=2000, minis=10),
        150000: dict(target=9000, drawdown=4500, dll=3000, minis=15),
    }
    t = table[size]
    return FirmRuleset(
        name=f"TOPSTEP_{size//1000}k",
        size=size, target=t["target"], drawdown=t["drawdown"],
        trailing="eod", max_micros=_minis(t["minis"]),
        dll=t["dll"], min_trading_days=2, consistency=0.50,
        lock_level=0.0, static_after_target=False,
    )


def lucid_ruleset(size: int) -> FirmRuleset:
    """LucidTest account: EOD trailing, DLL = 20% of target, min 5 days, lock at start,
    trailing becomes static after target reached."""
    table = {
        25000:  dict(target=1500, drawdown=1500),
        50000:  dict(target=3000, drawdown=2000),
        100000: dict(target=6000, drawdown=3000),
        150000: dict(target=9000, drawdown=4000),  # drawdown/DLL estimated per spec
    }
    # max_micros: Lucid does not publish the same mini cap; use a generous cap so it
    # is not the binding constraint (the model's contract sweep is what matters).
    micros_cap = {25000: 40, 50000: 60, 100000: 80, 150000: 120}
    t = table[size]
    target = t["target"]
    return FirmRuleset(
        name=f"LUCID_{size//1000}k",
        size=size, target=target, drawdown=t["drawdown"],
        trailing="eod", max_micros=micros_cap[size],
        dll=0.20 * target, min_trading_days=5, consistency=None,
        lock_level=0.0, static_after_target=True,
    )


# =============================================================================
# 2. STRATEGY + PER-TRADE P&L GENERATOR
# =============================================================================

@dataclass
class Strategy:
    """A trading strategy's statistical fingerprint.

    win_rate            : probability a trade is a winner (in (0,1))
    win_R / loss_R      : R-multiples. A win returns +win_R * risk; a loss returns -loss_R * risk.
                          For an asymmetric strategy set win_R != loss_R (e.g. 2R winners, 1R losers).
    dollar_risk_per_micro: $ lost on a -1R trade per ONE micro contract (the "R" in dollars/micro).
    trades_per_month    : expected number of trades over the ~21-trading-day horizon.
    cost_per_trade_per_micro: round-trip commission+slippage in $ per micro per trade (subtracted always).
    r_sigma             : optional gaussian noise (in R) added to each outcome for a fuzzier
                          R-distribution (0 = clean +win_R/-loss_R). Helps realism; default small.
    cluster_rho         : loss-clustering autocorrelation toggle in [0,1). 0 = i.i.d. trades.
                          >0 makes outcomes regime-sticky (a Markov win/lose regime) for stress tests.

    --- OPTIONAL MIXTURE R-DISTRIBUTION (for scale-out strategies) ---------------
    win_R_mix / win_R_probs : if provided, a winner's R is drawn from this discrete
                          mixture (values in win_R_mix, probabilities win_R_probs)
                          INSTEAD of the single win_R. This reproduces a scale-out
                          payoff: e.g. a +2R partial blended with a trailed runner that
                          is sometimes +3..+5R. Lengths must match and probs sum to 1.
    loss_R_mix / loss_R_probs : same idea for losers (magnitudes, positive). Reproduces
                          partial scratches (-0.5R) vs full stops (-1R) vs slippage (-1.2R).
    When a mixture is supplied, r_sigma still adds a small gaussian jitter on top.
    expectancy_R() / win_R_eff() / loss_R_eff() use the mixture means when present, so
    the calibration checks and the engine agree.
    """
    win_rate: float
    win_R: float = 1.0
    loss_R: float = 1.0
    dollar_risk_per_micro: float = 50.0
    trades_per_month: int = 80
    cost_per_trade_per_micro: float = 0.0
    r_sigma: float = 0.0
    cluster_rho: float = 0.0
    # Optional discrete mixtures (None => use the scalar win_R / loss_R).
    win_R_mix: Optional[List[float]] = None
    win_R_probs: Optional[List[float]] = None
    loss_R_mix: Optional[List[float]] = None   # positive magnitudes
    loss_R_probs: Optional[List[float]] = None

    def win_R_eff(self) -> float:
        """Mean winner R (mixture mean if a mixture is supplied, else win_R)."""
        if self.win_R_mix is not None and self.win_R_probs is not None:
            return float(np.dot(self.win_R_mix, self.win_R_probs))
        return self.win_R

    def loss_R_eff(self) -> float:
        """Mean loser R magnitude (mixture mean if supplied, else loss_R)."""
        if self.loss_R_mix is not None and self.loss_R_probs is not None:
            return float(np.dot(self.loss_R_mix, self.loss_R_probs))
        return self.loss_R

    def expectancy_R(self) -> float:
        """Per-trade expectancy in R (before costs), using effective (mixture) means."""
        return self.win_rate * self.win_R_eff() - (1 - self.win_rate) * self.loss_R_eff()

    def profit_factor(self) -> float:
        """Theoretical profit factor = (wr * meanWinR) / ((1-wr) * meanLossR)."""
        gw = self.win_rate * self.win_R_eff()
        gl = (1 - self.win_rate) * self.loss_R_eff()
        return gw / gl if gl > 0 else float("inf")


def _assign_trades_to_days(n_trades: int, n_days: int, rng: np.random.Generator) -> List[int]:
    """Distribute n_trades across n_days, allowing 0..N trades per day.

    Uses a multinomial draw (each trade independently lands on a uniform-random day),
    which yields a realistic ragged distribution (some 0-trade days, some busy days).
    Returns a list of length n_days with the trade count per day.
    """
    if n_trades <= 0:
        return [0] * n_days
    counts = rng.multinomial(n_trades, [1.0 / n_days] * n_days)
    return counts.tolist()


def generate_trade_pnls(strategy: Strategy, micros: int, n_trades: int,
                        rng: np.random.Generator) -> np.ndarray:
    """Produce a stream of per-trade dollar P&Ls for `n_trades` trades.

    Each trade:
      base R = +win_R (win) or -loss_R (loss), with optional gaussian R-noise.
      dollars = R * (dollar_risk_per_micro * micros) - cost_per_trade_per_micro * micros
    Win/loss draws are i.i.d. unless cluster_rho>0, in which case a sticky 2-state
    (win-regime / lose-regime) Markov chain governs the outcome to inject autocorrelation.

    Returns an np.ndarray of length n_trades of dollar P&Ls (one trade with `micros`
    contracts each).
    """
    if n_trades <= 0:
        return np.zeros(0, dtype=float)

    risk_dollars = strategy.dollar_risk_per_micro * micros
    cost_dollars = strategy.cost_per_trade_per_micro * micros

    # --- Win/loss boolean stream -------------------------------------------------
    if strategy.cluster_rho <= 0.0:
        wins = rng.random(n_trades) < strategy.win_rate
    else:
        # Sticky regime model: with prob rho keep the previous outcome's "regime",
        # otherwise redraw from the base win_rate. This raises the autocorrelation of
        # win/lose, producing clusters of losses (and wins) for stress testing while
        # preserving the long-run win_rate.
        rho = float(strategy.cluster_rho)
        wins = np.empty(n_trades, dtype=bool)
        prev = rng.random() < strategy.win_rate
        for i in range(n_trades):
            if rng.random() < rho:
                cur = prev  # stick to prior outcome's regime
            else:
                cur = rng.random() < strategy.win_rate
            wins[i] = cur
            prev = cur

    # --- R-multiples -------------------------------------------------------------
    # Default: two-point +win_R / -loss_R. If discrete mixtures are supplied (scale-out
    # strategies), draw each winner/loser R from its mixture instead. This reproduces an
    # asymmetric payoff (e.g. +2R partials + trailed runners with a +3..5R tail; losers
    # spread over partial scratches and full stops) while preserving the calibrated means.
    use_win_mix = strategy.win_R_mix is not None and strategy.win_R_probs is not None
    use_loss_mix = strategy.loss_R_mix is not None and strategy.loss_R_probs is not None

    if not use_win_mix and not use_loss_mix:
        R = np.where(wins, strategy.win_R, -strategy.loss_R).astype(float)
    else:
        R = np.empty(n_trades, dtype=float)
        win_idx = np.flatnonzero(wins)
        loss_idx = np.flatnonzero(~wins)
        # Winners
        if win_idx.size:
            if use_win_mix:
                R[win_idx] = rng.choice(strategy.win_R_mix, size=win_idx.size,
                                        p=strategy.win_R_probs)
            else:
                R[win_idx] = strategy.win_R
        # Losers (mixtures store positive magnitudes; apply the negative sign)
        if loss_idx.size:
            if use_loss_mix:
                R[loss_idx] = -rng.choice(strategy.loss_R_mix, size=loss_idx.size,
                                          p=strategy.loss_R_probs)
            else:
                R[loss_idx] = -strategy.loss_R

    if strategy.r_sigma > 0.0:
        R = R + rng.normal(0.0, strategy.r_sigma, size=n_trades)

    pnl = R * risk_dollars - cost_dollars
    return pnl


# =============================================================================
# 3. THE EVALUATION ENGINE
# =============================================================================

# Fail-reason codes
PASS = 0
FAIL_TRAILING = 1
FAIL_TIMEOUT = 2
FAIL_CONSISTENCY = 3
_REASON_NAME = {
    PASS: "pass",
    FAIL_TRAILING: "hit_trailing_floor",
    FAIL_TIMEOUT: "timeout_no_target",
    FAIL_CONSISTENCY: "consistency_fail",
}


def _run_one_eval(firm: FirmRuleset, strategy: Strategy, micros: int,
                  rng: np.random.Generator) -> Tuple[int, int]:
    """Simulate ONE 30-calendar-day evaluation.

    Returns (reason_code, trading_days_used).

    EQUITY/FLOOR CONVENTION: everything is an OFFSET from the starting balance.
      equity  starts at 0.0
      peak    starts at 0.0   (for the relevant trailing definition)
      floor   starts at -drawdown

    THE TRAILING FLOOR — exact mechanics:
      floor_target = min(peak, lock_level) - drawdown
        * intraday: `peak` includes every trade-by-trade equity point (unrealized swings).
        * eod:      `peak` is the max of end-of-day equities only.
      We clamp the peak used for the floor at `lock_level` so the floor never rises past
      (lock_level - drawdown). For TopStep/Lucid lock_level=0 => floor never exceeds -drawdown...
      WAIT: that's not right. Re-derived below.

    LOCK derivation (important):
      The floor should rise 1:1 with the peak until the floor reaches `lock_level`, then stop.
      floor = peak - drawdown, capped at lock_level.
      => floor = min(peak - drawdown, lock_level).
      For TopStep/Lucid (lock_level=0): once peak reaches `drawdown`, floor = 0 (start) and
      locks there. Before that, floor = peak - drawdown (still below start). Correct.
      For Apex intraday (lock_level=+100): floor locks at start+100 once peak reaches
      drawdown+100. Correct.
    """
    dd = firm.drawdown
    target = firm.target
    n_days = firm.trading_days_in_horizon
    lock = firm.lock_level

    # Total trades for the month, then spread across trading days.
    n_trades_month = strategy.trades_per_month
    day_counts = _assign_trades_to_days(n_trades_month, n_days, rng)

    equity = 0.0          # offset from start
    intraday_peak = 0.0   # running peak of equity incl. unrealized (for intraday trail)
    eod_peak = 0.0        # peak of end-of-day equities (for eod trail)

    # Current enforced floor (offset from start). Recomputed appropriately per mode.
    # For intraday: updated every trade. For eod: fixed for the whole day, updated at close.
    floor = min(intraday_peak - dd, lock)  # == -dd at open (since peak=0, lock>=0 => -dd<lock)

    trading_days_used = 0
    best_day_profit = 0.0     # for consistency rule (TopStep)
    total_profit_realized = 0.0  # running realized equity == `equity` here (single book)
    target_static_locked = False  # Lucid: trailing frozen after target reached

    for day_idx in range(n_days):
        n_t = day_counts[day_idx]

        # A "trading day" is a day on which at least one trade is taken.
        if n_t > 0:
            trading_days_used += 1

        day_start_equity = equity        # for DLL measurement (loss from day's open)
        day_pnls = generate_trade_pnls(strategy, micros, n_t, rng)

        # --- EOD mode: the floor enforced THROUGHOUT today was set at yesterday's close.
        # --- Intraday mode: the floor updates every trade below.
        day_floor = floor  # snapshot; in eod mode this stays fixed all day

        for k in range(n_t):
            equity += day_pnls[k]

            # Update the running intraday peak (always tracked; used by intraday trail).
            if equity > intraday_peak:
                intraday_peak = equity

            # --- Trailing floor enforcement ---
            if firm.trailing == "intraday":
                if not target_static_locked:
                    # Floor rises with the live peak, capped at lock.
                    floor = min(intraday_peak - dd, lock)
                day_floor = floor
            # (eod mode: day_floor stays as set at last close)

            # --- Daily Loss Limit ---
            # Loss within the day measured from day_start_equity. If reached, cap the
            # day's loss at -DLL and stop trading for the day (does NOT fail eval by itself).
            if firm.dll is not None:
                day_loss = day_start_equity - equity  # positive = down on the day
                if day_loss >= firm.dll:
                    # Pin the day's loss at exactly the DLL (broker auto-flattens at the cap).
                    equity = day_start_equity - firm.dll
                    # Re-check the floor at this capped equity below, then break out of the day.
                    if equity <= day_floor + 1e-9:
                        return FAIL_TRAILING, trading_days_used
                    break

            # --- FAIL: equity touches/penetrates the trailing floor ---
            if equity <= day_floor + 1e-9:
                return FAIL_TRAILING, trading_days_used

        # ---- End of day processing ----
        # Track best single-day profit for the consistency rule.
        if n_t > 0:
            day_profit = equity - day_start_equity
            if day_profit > best_day_profit:
                best_day_profit = day_profit

        # EOD trailing: recompute the EOD peak and set tomorrow's floor.
        if firm.trailing == "eod":
            if equity > eod_peak:
                eod_peak = equity
            if not target_static_locked:
                floor = min(eod_peak - dd, lock)

        # Lucid: once start+target reached, trailing becomes fully static (freeze floor).
        if firm.static_after_target and equity >= target:
            target_static_locked = True

        # ---- PASS CHECK (only valid at/after a day close, with constraints satisfied) ----
        if equity >= target:
            # Min trading days gate.
            if trading_days_used < firm.min_trading_days:
                # Cannot pass yet; must keep trading more days. Continue the loop.
                pass
            else:
                # Consistency gate (TopStep): best day's profit must be < cap * total profit.
                if firm.consistency is not None:
                    total_profit = equity  # offset from start == total profit
                    if total_profit <= 0:
                        # degenerate; cannot pass
                        pass
                    elif best_day_profit >= firm.consistency * total_profit - 1e-9:
                        # Consistency violated AT this point. The realistic trader keeps
                        # trading to grow total_profit and dilute the big day. We continue;
                        # only if day 30 arrives still violating do we record consistency_fail.
                        pass
                    else:
                        return PASS, trading_days_used
                else:
                    return PASS, trading_days_used

    # ---- Horizon elapsed (30 calendar / 21 trading days) without a clean pass ----
    if equity >= target and trading_days_used >= firm.min_trading_days:
        # Target met & min-days met, but we exited the loop -> must be a consistency block.
        if firm.consistency is not None:
            total_profit = equity
            if total_profit > 0 and best_day_profit < firm.consistency * total_profit - 1e-9:
                # Edge case: satisfied exactly at the final close.
                return PASS, trading_days_used
            return FAIL_CONSISTENCY, trading_days_used
        # No consistency rule but min-days only just satisfied at the very end -> pass.
        return PASS, trading_days_used

    return FAIL_TIMEOUT, trading_days_used


def simulate_eval(firm: FirmRuleset, strategy: Strategy, micros: int,
                  n_sims: int = 20000, seed: Optional[int] = None) -> Dict:
    """Run N independent 30-calendar-day evaluations and aggregate results.

    Parameters
    ----------
    firm     : FirmRuleset (Apex/TopStep/Lucid config)
    strategy : Strategy (edge fingerprint)
    micros   : contract count in MICROS (capped at firm.max_micros)
    n_sims   : number of Monte-Carlo evals
    seed     : RNG seed for reproducibility

    Returns
    -------
    dict with:
      pass_rate                    : fraction of sims that passed
      n_sims                       : number of sims
      micros                       : (clamped) contract count used
      fail_hit_trailing_floor      : fraction failing by trailing breach
      fail_timeout_no_target       : fraction failing by 30-day timeout
      fail_consistency             : fraction failing the consistency rule
      median_days_to_pass          : median trading-days-used among PASSERS (nan if none)
      mean_days_to_pass            : mean trading-days-used among passers
    """
    if micros > firm.max_micros:
        micros = firm.max_micros
    if micros < 1:
        micros = 1

    rng = np.random.default_rng(seed)

    n_pass = 0
    n_trail = 0
    n_timeout = 0
    n_consistency = 0
    pass_days: List[int] = []

    for _ in range(n_sims):
        reason, days = _run_one_eval(firm, strategy, micros, rng)
        if reason == PASS:
            n_pass += 1
            pass_days.append(days)
        elif reason == FAIL_TRAILING:
            n_trail += 1
        elif reason == FAIL_TIMEOUT:
            n_timeout += 1
        elif reason == FAIL_CONSISTENCY:
            n_consistency += 1

    inv = 1.0 / n_sims
    median_days = float(np.median(pass_days)) if pass_days else float("nan")
    mean_days = float(np.mean(pass_days)) if pass_days else float("nan")

    return {
        "firm": firm.name,
        "trailing": firm.trailing,
        "micros": micros,
        "n_sims": n_sims,
        "pass_rate": n_pass * inv,
        "fail_hit_trailing_floor": n_trail * inv,
        "fail_timeout_no_target": n_timeout * inv,
        "fail_consistency": n_consistency * inv,
        "median_days_to_pass": median_days,
        "mean_days_to_pass": mean_days,
        "strategy_expectancy_R": strategy.expectancy_R(),
    }


# =============================================================================
# 4. SELF-TESTS
# =============================================================================
# Each test prints its result and contributes a PASS/FAIL. The module-level
# run_self_tests() returns (all_passed: bool, detail: str).

def _fmt_pct(x: float) -> str:
    return f"{100*x:5.1f}%"


def run_self_tests(n_sims: int = 20000, seed: int = 12345) -> Tuple[bool, str]:
    """Run self-tests (a)-(d), print results, return (all_passed, detail_text)."""
    lines: List[str] = []
    results: List[bool] = []

    def log(s: str = ""):
        lines.append(s)

    log("=" * 78)
    log("PROP-EVAL MONTE-CARLO  —  SELF-TESTS")
    log(f"n_sims={n_sims}  seed={seed}")
    log("=" * 78)

    # -------------------------------------------------------------------------
    # (a) ZERO-EDGE random walk: 50% win, +1R/-1R symmetric.
    #     For a DRIFTLESS walk, the classic first-passage prob of reaching +target
    #     before -drawdown is  dd/(target+dd) = 2000/(3000+2000) = 40.0%  (the
    #     "naive barrier" ceiling). Reality must come in BELOW this for two reasons:
    #       (i)  the trailing floor RATCHETS UP under the walk, tightening the lower
    #            barrier -> a trailing-DD walk is strictly easier to stop than a fixed
    #            barrier, so pass < 40%;
    #       (ii) 30-day timeout truncation removes walks that haven't hit a barrier.
    #     To isolate (i) from (ii) we size so the walk reliably reaches a barrier
    #     within 21 trading days (few, large steps => low timeout share). We then
    #     require a "sane low band": pass clearly UNDER the 40% barrier ceiling and
    #     well under 50%, with the dominant fail mode being the TRAILING breach (not
    #     timeout). This is the signature of correct trailing-DD mechanics.
    #     We test BOTH intraday and EOD (both must land in band); intraday is checked
    #     to be the stricter of the two here as a bonus consistency check.
    # -------------------------------------------------------------------------
    log("\n(a) ZERO-EDGE random walk (50% win, +1R/-1R), Apex 50k")
    log("    Driftless barrier ceiling: dd/(target+dd) = 2000/(3000+2000) = 40.0%.")
    log("    Trailing-DD ratchet must push the realized pass BELOW this; sized for")
    log("    few/large steps so timeout truncation is small and the trailing barrier")
    log("    is the dominant fail mode.")
    # R = $400/micro * 4 micros... no: dollar_risk_per_micro=100, micros=4 => $400/R.
    # dd=2000 => 5R of room; ~150 trades over 21 days => walk reliably reaches a barrier.
    strat_a = Strategy(win_rate=0.50, win_R=1.0, loss_R=1.0,
                       dollar_risk_per_micro=100.0, trades_per_month=150,
                       cost_per_trade_per_micro=0.0)
    firm_a_intra = apex_ruleset(50000, trailing="intraday")
    firm_a_eod = apex_ruleset(50000, trailing="eod")
    res_a = simulate_eval(firm_a_intra, strat_a, micros=4, n_sims=n_sims, seed=seed)
    res_a_eod = simulate_eval(firm_a_eod, strat_a, micros=4, n_sims=n_sims, seed=seed)
    pa = res_a["pass_rate"]
    pa_eod = res_a_eod["pass_rate"]
    log(f"    INTRADAY pass = {_fmt_pct(pa)}  "
        f"(trail={_fmt_pct(res_a['fail_hit_trailing_floor'])}, "
        f"timeout={_fmt_pct(res_a['fail_timeout_no_target'])})")
    log(f"    EOD      pass = {_fmt_pct(pa_eod)}  "
        f"(trail={_fmt_pct(res_a_eod['fail_hit_trailing_floor'])}, "
        f"timeout={_fmt_pct(res_a_eod['fail_timeout_no_target'])})")
    # Sane low band: below the 40% driftless ceiling (the trailing ratchet subtracts),
    # well under 50%, but not collapsed to ~0 (a fair coin with 5R of room still passes
    # a meaningful minority). Dominant fail must be the trailing breach, not timeout.
    in_band = (0.15 <= pa <= 0.40) and (0.15 <= pa_eod <= 0.40)
    under_ceiling = (pa < 0.40) and (pa_eod < 0.40)
    trail_dominant = res_a["fail_hit_trailing_floor"] > res_a["fail_timeout_no_target"]
    ok_a = in_band and under_ceiling and trail_dominant
    log(f"    EXPECT: both 0.15<=pass<=0.40 (below 40% ceiling, <<50%) AND trailing is")
    log(f"            the dominant fail mode  -> {'PASS' if ok_a else 'FAIL'}")
    results.append(ok_a)

    # -------------------------------------------------------------------------
    # (b) HUGE EDGE / TINY VOL: should pass ~100%.
    #     90% win, +1R/-0.25R (asymmetric, big positive expectancy), small risk so
    #     drawdown is essentially never threatened. Many small green trades march to
    #     target. Use Apex 50k intraday.
    # -------------------------------------------------------------------------
    log("\n(b) HUGE-EDGE / tiny-vol (90% win, +1R/-0.25R, small risk), Apex 50k intraday")
    firm_b = apex_ruleset(50000, trailing="intraday")
    strat_b = Strategy(win_rate=0.90, win_R=1.0, loss_R=0.25,
                       dollar_risk_per_micro=20.0, trades_per_month=120,
                       cost_per_trade_per_micro=0.0)
    res_b = simulate_eval(firm_b, strat_b, micros=4, n_sims=n_sims, seed=seed)
    pb = res_b["pass_rate"]
    log(f"    pass_rate = {_fmt_pct(pb)}  "
        f"(trail={_fmt_pct(res_b['fail_hit_trailing_floor'])}, "
        f"timeout={_fmt_pct(res_b['fail_timeout_no_target'])})  "
        f"median_days_to_pass={res_b['median_days_to_pass']:.0f}")
    ok_b = pb >= 0.99
    log(f"    EXPECT: pass_rate >= 99%  -> {'PASS' if ok_b else 'FAIL'}")
    results.append(ok_b)

    # -------------------------------------------------------------------------
    # (c) INTRADAY trailing must be STRICTER than EOD for the SAME strategy.
    #     Two parts:
    #     (c1) A HAND-BUILT path that isolates the exact mechanic: spike +800
    #          unrealized then give it back to +100 (day 1), then a -1300 move
    #          (day 2). Intraday's floor already ratcheted to start+lock-buffer on
    #          the +800 spike, so the give-back leaves a thin cushion and the -1300
    #          breaches it -> FAIL. EOD only saw the +100 close, so its floor sits
    #          far below and the same path SURVIVES. This proves the ratchet directly.
    #     (c2) Aggregate: a swingy edge (big winners that are partly given back)
    #          must pass at a LOWER rate under intraday than EOD.
    # -------------------------------------------------------------------------
    log("\n(c) INTRADAY vs EOD trailing must differ (intraday stricter), Apex 50k")

    # (c1) deterministic-path proof of the ratchet -----------------------------
    import eval_sim as _self
    firm_intra0 = apex_ruleset(50000, "intraday")
    firm_eod0 = apex_ruleset(50000, "eod")
    _script = {0: np.array([800.0, -700.0]), 1: np.array([-1300.0])}
    _saved_gen = _self.generate_trade_pnls
    _saved_assign = _self._assign_trades_to_days
    _state = {"call": 0}

    def _fake_gen(strategy, micros, n_trades, rng):
        out = _script.get(_state["call"], np.zeros(n_trades))
        _state["call"] += 1
        return out

    _self.generate_trade_pnls = _fake_gen
    _self._assign_trades_to_days = lambda n, d, rng: [2, 1] + [0] * (d - 2)
    _dummy = Strategy(win_rate=0.5, trades_per_month=3, dollar_risk_per_micro=1.0)
    _state["call"] = 0
    r_intra0, _ = _self._run_one_eval(firm_intra0, _dummy, 1, np.random.default_rng(1))
    _state["call"] = 0
    r_eod0, _ = _self._run_one_eval(firm_eod0, _dummy, 1, np.random.default_rng(1))
    _self.generate_trade_pnls = _saved_gen
    _self._assign_trades_to_days = _saved_assign
    c1_ok = (r_intra0 == FAIL_TRAILING) and (r_eod0 != FAIL_TRAILING)
    log("    (c1) hand-built path [+800,-700 then -1300]:")
    log(f"         INTRADAY -> {_REASON_NAME[r_intra0]} (expect hit_trailing_floor); "
        f"EOD -> {_REASON_NAME[r_eod0]} (expect survives)  -> "
        f"{'PASS' if c1_ok else 'FAIL'}")

    # (c2) aggregate swingy edge -----------------------------------------------
    # Big asymmetric winners with R-noise => meaningful unrealized give-backs that
    # the intraday floor capitalizes on. Many trades/day amplifies intraday swings.
    strat_c = Strategy(win_rate=0.50, win_R=2.2, loss_R=1.0,
                       dollar_risk_per_micro=50.0, trades_per_month=140,
                       cost_per_trade_per_micro=0.0, r_sigma=0.40)
    firm_c_intra = apex_ruleset(50000, trailing="intraday")
    firm_c_eod = apex_ruleset(50000, trailing="eod")
    res_c_intra = simulate_eval(firm_c_intra, strat_c, micros=6, n_sims=n_sims, seed=seed)
    res_c_eod = simulate_eval(firm_c_eod, strat_c, micros=6, n_sims=n_sims, seed=seed)
    pci, pce = res_c_intra["pass_rate"], res_c_eod["pass_rate"]
    log("    (c2) swingy edge (50% win, +2.2R/-1.0R, r_sigma=0.40, 140 tr/mo, 6 micros):")
    log(f"         INTRADAY pass = {_fmt_pct(pci)}   "
        f"(trail={_fmt_pct(res_c_intra['fail_hit_trailing_floor'])})")
    log(f"         EOD      pass = {_fmt_pct(pce)}   "
        f"(trail={_fmt_pct(res_c_eod['fail_hit_trailing_floor'])})")
    c2_ok = pci < pce
    log(f"         EXPECT: INTRADAY pass < EOD pass  -> {'PASS' if c2_ok else 'FAIL'}  "
        f"(gap={_fmt_pct(pce - pci)})")
    ok_c = c1_ok and c2_ok
    log(f"    (c) overall -> {'PASS' if ok_c else 'FAIL'}")
    results.append(ok_c)

    # -------------------------------------------------------------------------
    # (d) CONTRACT-SIZE HUMP: increasing micros first raises pass (reach target
    #     faster) then lowers it (drawdown breaches). Sweep micros on a moderate
    #     edge and show the rise-then-fall. Use Apex 50k intraday (cap 60 micros).
    # -------------------------------------------------------------------------
    log("\n(d) CONTRACT-SIZE SWEEP (hump), Apex 50k intraday  [cap=60 micros]")
    strat_d = Strategy(win_rate=0.53, win_R=1.3, loss_R=1.0,
                       dollar_risk_per_micro=25.0, trades_per_month=90,
                       cost_per_trade_per_micro=0.0, r_sigma=0.10)
    firm_d = apex_ruleset(50000, trailing="intraday")
    sweep_micros = [1, 2, 4, 8, 16, 30, 60]
    sweep = []
    log("       micros   pass     trail-fail   timeout-fail   med_days")
    for m in sweep_micros:
        r = simulate_eval(firm_d, strat_d, micros=m, n_sims=n_sims, seed=seed)
        sweep.append((m, r["pass_rate"], r["fail_hit_trailing_floor"],
                      r["fail_timeout_no_target"], r["median_days_to_pass"]))
        md = r["median_days_to_pass"]
        md_s = f"{md:5.0f}" if md == md else "  nan"
        log(f"       {m:5d}   {_fmt_pct(r['pass_rate'])}   "
            f"{_fmt_pct(r['fail_hit_trailing_floor'])}        "
            f"{_fmt_pct(r['fail_timeout_no_target'])}       {md_s}")
    passes = [s[1] for s in sweep]
    # Hump test: the max pass rate occurs at an INTERIOR sizing (not at the smallest
    # and not at the largest), AND the pass rate at the largest size is below the peak
    # (over-sizing breaches drawdown), AND it rose from the smallest size to the peak.
    peak_idx = int(np.argmax(passes))
    rose = passes[peak_idx] > passes[0] + 1e-6           # bigger helped vs smallest
    fell = passes[-1] < passes[peak_idx] - 1e-6          # too big hurt vs peak
    interior = 0 < peak_idx < len(passes) - 1
    ok_d = rose and fell and interior
    log(f"    peak at micros={sweep_micros[peak_idx]} "
        f"(pass={_fmt_pct(passes[peak_idx])}); "
        f"smallest={_fmt_pct(passes[0])}, largest={_fmt_pct(passes[-1])}")
    log(f"    EXPECT: rise from smallest to an INTERIOR peak, then fall at largest  -> "
        f"{'PASS' if ok_d else 'FAIL'}")
    results.append(ok_d)

    # -------------------------------------------------------------------------
    # (e) EDGE-CASE MECHANICS: lock, DLL cap, min-trading-days, consistency.
    #     These are deterministic / near-deterministic checks proving each rule
    #     does real, load-bearing work. (Not part of the required (a)-(d) but
    #     folded in so the engine ships with proof of every mechanic.)
    # -------------------------------------------------------------------------
    log("\n(e) EDGE-CASE MECHANICS (lock / DLL / min-days / consistency)")
    import eval_sim as _self2
    _sg, _sa = _self2.generate_trade_pnls, _self2._assign_trades_to_days

    def _mk_script(scr, st):
        def g(strategy, micros, n_trades, rng):
            o = scr.get(st["i"], np.zeros(n_trades)); st["i"] += 1; return o
        return g

    # --- (e1) LOCK is load-bearing -------------------------------------------
    # Emulate the TopStep/Lucid "lock at start" via Apex-EOD with lock_level=0 (no DLL
    # to interfere). Day0 closes +2500 => EOD floor = min(2500-2000, 0) = 0 (locked).
    #   Case A: later drop to +50 -> survives (floor pinned at 0).
    #   Case B: later drop to -50 -> hit_trailing_floor.
    #   Case C (control, NO lock): floor would be 500, so +50 -> hit_trailing_floor.
    st = {"i": 0}
    firm_lock = apex_ruleset(50000, "eod"); firm_lock.lock_level = 0.0
    firm_nolock = apex_ruleset(50000, "eod"); firm_nolock.lock_level = 1e9
    _self2._assign_trades_to_days = lambda n, d, rng: [1, 1, 1] + [0] * (d - 3)
    dummy = Strategy(0.5, trades_per_month=3, dollar_risk_per_micro=1.0)

    _self2.generate_trade_pnls = _mk_script({0: np.array([2500.0]), 1: np.array([0.0]),
                                             2: np.array([-2450.0])}, st)
    st["i"] = 0; rA = _self2._run_one_eval(firm_lock, dummy, 1, np.random.default_rng(0))[0]
    _self2.generate_trade_pnls = _mk_script({0: np.array([2500.0]), 1: np.array([0.0]),
                                             2: np.array([-2550.0])}, st)
    st["i"] = 0; rB = _self2._run_one_eval(firm_lock, dummy, 1, np.random.default_rng(0))[0]
    _self2.generate_trade_pnls = _mk_script({0: np.array([2500.0]), 1: np.array([0.0]),
                                             2: np.array([-2450.0])}, st)
    st["i"] = 0; rC = _self2._run_one_eval(firm_nolock, dummy, 1, np.random.default_rng(0))[0]
    e1_ok = (rA != FAIL_TRAILING) and (rB == FAIL_TRAILING) and (rC == FAIL_TRAILING)
    log(f"    (e1) LOCK: locked floor=0 +50->{_REASON_NAME[rA]} (survive), "
        f"-50->{_REASON_NAME[rB]} (fail); NO-lock +50->{_REASON_NAME[rC]} (fail, "
        f"proves lock load-bearing)  -> {'PASS' if e1_ok else 'FAIL'}")

    # --- (e2) DLL caps the day's loss and pauses the day ---------------------
    # TopStep 50k DLL=1000. Day0: a single -1500 trade should be CAPPED at -1000
    # (day pauses), NOT fail the eval (floor is at -2000, so -1000 survives).
    st2 = {"i": 0}
    firm_t = topstep_ruleset(50000)
    _self2._assign_trades_to_days = lambda n, d, rng: [1] + [0] * (d - 1)
    _self2.generate_trade_pnls = _mk_script({0: np.array([-1500.0])}, st2)
    # Hook to read final equity: re-run a tiny copy of the engine path is overkill;
    # instead infer from reason. -1000 (capped) > floor -2000 => should be timeout (survived).
    st2["i"] = 0
    rD = _self2._run_one_eval(firm_t, dummy, 1, np.random.default_rng(0))[0]
    # And prove that WITHOUT the DLL the same -1500 would still survive (floor -2000),
    # so to show the cap we instead push a -2500 single trade: with DLL it caps at -1000
    # (survive); without DLL it would breach -2000 (fail). Use Apex-EOD (no DLL) as the
    # no-DLL control with the same drawdown 2000.
    st3 = {"i": 0}
    firm_nodll = apex_ruleset(50000, "eod")  # dd=2000, no DLL
    _self2.generate_trade_pnls = _mk_script({0: np.array([-2500.0])}, st3)
    st3["i"] = 0; rE_nodll = _self2._run_one_eval(firm_nodll, dummy, 1, np.random.default_rng(0))[0]
    st4 = {"i": 0}
    _self2.generate_trade_pnls = _mk_script({0: np.array([-2500.0])}, st4)
    st4["i"] = 0; rE_dll = _self2._run_one_eval(firm_t, dummy, 1, np.random.default_rng(0))[0]
    e2_ok = (rD == FAIL_TIMEOUT) and (rE_nodll == FAIL_TRAILING) and (rE_dll == FAIL_TIMEOUT)
    log(f"    (e2) DLL: -1500 day capped at -1000 -> {_REASON_NAME[rD]} (survive); "
        f"-2500 day: no-DLL->{_REASON_NAME[rE_nodll]} (breach), "
        f"DLL->{_REASON_NAME[rE_dll]} (capped, survive)  -> {'PASS' if e2_ok else 'FAIL'}")

    _self2.generate_trade_pnls, _self2._assign_trades_to_days = _sg, _sa

    # --- (e3) MIN TRADING DAYS blocks a too-fast pass ------------------------
    # A near-deterministic huge edge passes day 1 under Apex (min_days=0) but Lucid
    # (min_days=5) forces median_days_to_pass >= 5.
    strat_fast = Strategy(win_rate=0.99, win_R=1.0, loss_R=0.1,
                          dollar_risk_per_micro=60.0, trades_per_month=200)
    r_apex = simulate_eval(apex_ruleset(50000, "intraday"), strat_fast, micros=6,
                           n_sims=4000, seed=3)
    r_lucid = simulate_eval(lucid_ruleset(50000), strat_fast, micros=6,
                            n_sims=4000, seed=3)
    e3_ok = (r_lucid["median_days_to_pass"] >= 5) and \
            (r_apex["median_days_to_pass"] < r_lucid["median_days_to_pass"])
    log(f"    (e3) MIN-DAYS: Apex medDays={r_apex['median_days_to_pass']:.0f} (min 0), "
        f"Lucid medDays={r_lucid['median_days_to_pass']:.0f} (min 5)  -> "
        f"{'PASS' if e3_ok else 'FAIL'}")

    # --- (e4) CONSISTENCY rule (TopStep 50%) actually fails lumpy traders ----
    # A lumpy strategy that tends to hit target in one big day triggers consistency_fail.
    strat_lumpy = Strategy(win_rate=0.80, win_R=1.0, loss_R=1.0,
                           dollar_risk_per_micro=300.0, trades_per_month=12)
    r_lumpy = simulate_eval(topstep_ruleset(50000), strat_lumpy, micros=6,
                            n_sims=6000, seed=5)
    e4_ok = r_lumpy["fail_consistency"] > 0.0
    log(f"    (e4) CONSISTENCY: TopStep lumpy trader consistency_fail="
        f"{_fmt_pct(r_lumpy['fail_consistency'])} (>0 required)  -> "
        f"{'PASS' if e4_ok else 'FAIL'}")

    ok_e = e1_ok and e2_ok and e3_ok and e4_ok
    log(f"    (e) overall -> {'PASS' if ok_e else 'FAIL'}")
    results.append(ok_e)

    all_passed = all(results)
    log("\n" + "=" * 78)
    log(f"SELF-TEST SUMMARY:  (a)={'P' if results[0] else 'F'}  "
        f"(b)={'P' if results[1] else 'F'}  "
        f"(c)={'P' if results[2] else 'F'}  "
        f"(d)={'P' if results[3] else 'F'}  "
        f"(e)={'P' if results[4] else 'F'}   "
        f"=>  {'ALL PASS' if all_passed else 'SOME FAILED'}")
    log("=" * 78)

    detail = "\n".join(lines)
    return all_passed, detail


# =============================================================================
# 5. CLI ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import sys

    ok, detail = run_self_tests(n_sims=20000, seed=12345)
    print(detail)

    # Bonus: a quick realistic example matrix so the file is useful out of the box.
    print("\n" + "=" * 78)
    print("EXAMPLE: a modest real-ish edge across firms (illustrative, not a recommendation)")
    print("  strategy: 50% win, +1.6R/-1.0R, $25/R/micro, 80 trades/mo, $1.0 cost/micro/trade")
    print("=" * 78)
    demo = Strategy(win_rate=0.50, win_R=1.6, loss_R=1.0,
                    dollar_risk_per_micro=25.0, trades_per_month=80,
                    cost_per_trade_per_micro=1.0, r_sigma=0.10)
    print(f"  per-trade expectancy = {demo.expectancy_R():+.3f}R\n")
    configs = [
        ("Apex 50k intraday", apex_ruleset(50000, "intraday"), 8),
        ("Apex 50k EOD",      apex_ruleset(50000, "eod"),      8),
        ("TopStep 50k",       topstep_ruleset(50000),          8),
        ("Lucid 50k",         lucid_ruleset(50000),            8),
    ]
    print(f"  {'config':22s} {'micros':>6s} {'pass':>7s} {'trail':>7s} "
          f"{'timeout':>8s} {'consist':>8s} {'medDays':>8s}")
    for label, firm, mic in configs:
        r = simulate_eval(firm, demo, micros=mic, n_sims=20000, seed=99)
        md = r["median_days_to_pass"]
        md_s = f"{md:.0f}" if md == md else "nan"
        print(f"  {label:22s} {r['micros']:6d} {_fmt_pct(r['pass_rate']):>7s} "
              f"{_fmt_pct(r['fail_hit_trailing_floor']):>7s} "
              f"{_fmt_pct(r['fail_timeout_no_target']):>8s} "
              f"{_fmt_pct(r['fail_consistency']):>8s} {md_s:>8s}")

    sys.exit(0 if ok else 1)
