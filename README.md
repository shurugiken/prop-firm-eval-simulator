# Prop-Firm Eval Simulator

A Monte-Carlo simulator that estimates the **honest probability of passing a futures prop-firm evaluation** (Apex / TopStep / Lucid) within 30 days — and, just as importantly, exposes *why* the headline pass rates quoted around these evals are usually wrong.

> **The one-line takeaway:** a high simulated pass rate (75–96%) is almost always an **oversizing + idealized daily-loss-limit artifact**, not evidence of an edge. Model the daily-loss-limit the way a real stop actually fills and those numbers collapse toward ~50%. This repo is built to make that artifact visible.

## Why this exists

Prop-firm "combine" math is dominated by one tricky mechanic: the **trailing drawdown**. Most back-of-envelope estimates (and most marketing) get it wrong in the optimistic direction. Two failure modes in particular inflate the apparent pass rate:

1. **Trailing-drawdown ratchet** — an *intraday* trailing floor rises with every unrealized equity spike and never comes back down, permanently eroding your cushion. A naive "barrier" model that ignores the ratchet over-counts survivors.
2. **The daily-loss-limit (DLL) idealization** — if you assume every losing day is pinned to *exactly* `−$DLL` (a perfect broker auto-flatten), you can "pass" by sizing absurdly large (e.g. a single stop = 3× the entire drawdown), because the model magically caps each bad day. A real atomic stop-loss fills in full *before* any auto-flatten reacts. Correct for that and the pass rate falls off a cliff.

The tell-tale of a broken model: **pass rate rising monotonically with contract size**. A correct model has a *hump* — an interior optimal size — because beyond it, one bad stop blows the floor.

## What it models (verified, June 2026 rulesets — re-verify at signup; firms change these)

| Firm | Trailing | Daily loss limit | Min days | Consistency | Notes |
|---|---|---|---|---|---|
| **Apex 4.0** | choose EOD or intraday | off by default | 0 | none | 30 calendar days; EOD is gentler |
| **TopStep** | EOD | enforced | ~2 (via consistency) | best day < 50% of total | |
| **Lucid** | EOD | 20% of target | 5 | none | trailing locks to static once target reached; 25k = 1:1 = most lenient |

It models the exact trailing-floor mechanics (intraday ratchet vs. EOD), the lock level, the DLL day-cap, the minimum-trading-days gate, the consistency rule, and a 30-day timeout — then runs N independent evaluations and reports the pass rate **and the breakdown of how failures happen** (hit the floor / ran out of time / failed consistency).

## Install

```bash
pip install -r requirements.txt   # numpy only
```

## Usage

```python
from eval_sim import Strategy, lucid_ruleset, apex_ruleset, simulate_eval

# Describe a strategy by its statistical fingerprint (no strategy logic needed):
strat = Strategy(
    win_rate=0.50, win_R=1.6, loss_R=1.0,   # 50% win, +1.6R winners / -1R losers
    dollar_risk_per_micro=25.0,             # $ lost on a -1R trade per ONE micro
    trades_per_month=80,
    cost_per_trade_per_micro=1.0,           # round-trip commission+slippage
)

# Sweep contract size to find the pass-maximizing config (note the hump):
for micros in [2, 4, 6, 8, 12, 20, 40]:
    r = simulate_eval(lucid_ruleset(25000), strat, micros=micros, n_sims=20000, seed=1)
    print(f"{micros:3d} micros  pass={r['pass_rate']:.1%}  "
          f"floor-fail={r['fail_hit_trailing_floor']:.1%}")
```

Run the file directly to execute the self-test suite plus an illustrative cross-firm example:

```bash
python eval_sim.py
```

## Self-tests (the mechanics are tested, not just asserted)

`python eval_sim.py` runs a battery that demonstrates each mechanic does real, load-bearing work:

- **(a)** a zero-edge coin-flip lands *below* the driftless barrier ceiling `dd/(target+dd)` — proving the trailing ratchet tightens the floor.
- **(b)** a huge, low-variance edge passes ~100%.
- **(c)** intraday trailing is strictly *stricter* than EOD (a hand-built `+800 then give-back then -1300` path fails intraday but survives EOD).
- **(d)** the **contract-size hump** — pass rate rises then falls as size grows.
- **(e)** lock / DLL cap / min-trading-days / consistency each change outcomes as designed.

## EV framing

Even at an honest ~50% pass rate, the eval fee behaves like a cheap, repeatable call option: expected fees to land one funded account ≈ `fee / pass_rate`. But that is "+EV *to acquire* a funded account," which is a **separate bet** from "+EV *trading* the funded account." Acquiring the account is only worth it if the underlying edge is real and **forward-proven first** — never pay an eval fee to monetize a strategy you haven't proven live.

## What this is *not*

It contains **no trading strategy and no edge** — it's a risk-mechanics calculator. You supply the statistical fingerprint (win rate, R-distribution, risk, frequency); it returns pass probability and failure modes. The firm rulesets are public information.

## License

MIT © Kwashawn Warren
