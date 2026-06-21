"""
test_eval_sim.py — pytest suite for eval_sim.py

Covers:
  - FirmRuleset builders and helper methods
  - Strategy expectancy / profit-factor helpers (scalar and mixture)
  - generate_trade_pnls: determinism, cost deduction, r_sigma noise, cluster_rho stickiness
  - _assign_trades_to_days: zero-trade edge case, sum invariant
  - simulate_eval: aggregate stats with fixed seed; micros clamping (lo + hi);
    fraction sum == 1.0 invariant; nan days when no pass; n_sims=1 works
  - Pass/fail rule evaluation:
      * huge edge => ~100% pass
      * near-certain-fail edge => 0% pass
      * zero edge => pass rate below the 40% driftless-barrier ceiling,
        trailing breach is the dominant fail mode
  - Intraday vs EOD trailing: hand-built path proves the ratchet;
    swingy aggregate strategy has lower pass rate intraday than EOD
  - DLL caps the day's loss and does not immediately fail the eval
  - Lock level: locked floor at 0 keeps a survivor alive; un-locked floor kills it
  - Min trading days: Lucid forces median_days_to_pass >= 5 even for a huge edge
  - Consistency rule: TopStep registers non-zero consistency_fail on a lumpy trader
  - Contract-size hump: pass rate rises then falls as micros increases

All tests use fixed RNG seeds and mock network/filesystem calls are not needed
(the module is pure computation). No real external calls are made.
"""

import math
import sys
import os

import numpy as np
import pytest

# Make the project root importable regardless of where pytest is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_sim as e
from eval_sim import (
    Strategy,
    FirmRuleset,
    apex_ruleset,
    topstep_ruleset,
    lucid_ruleset,
    generate_trade_pnls,
    _assign_trades_to_days,
    simulate_eval,
    _run_one_eval,
    PASS,
    FAIL_TRAILING,
    FAIL_TIMEOUT,
    FAIL_CONSISTENCY,
)


# =============================================================================
# 1. FirmRuleset builders — spot-check the hard-coded numbers
# =============================================================================

class TestFirmRulesetBuilders:

    def test_apex_initial_floor(self):
        """initial_floor() must equal -drawdown for every Apex size."""
        for size, expected_dd in [(25000, 1000), (50000, 2000), (100000, 3000), (150000, 4000)]:
            f = apex_ruleset(size, "intraday")
            assert f.initial_floor() == pytest.approx(-expected_dd)

    def test_apex_50k_intraday_ruleset(self):
        f = apex_ruleset(50000, "intraday")
        assert f.target == 3000
        assert f.drawdown == 2000
        assert f.max_micros == 60       # 6 minis * 10
        assert f.trailing == "intraday"
        assert f.dll is None
        assert f.min_trading_days == 0
        assert f.consistency is None
        assert f.lock_level == pytest.approx(100.0)  # default lock_buffer

    def test_apex_eod_trailing_field(self):
        f = apex_ruleset(50000, "eod")
        assert f.trailing == "eod"

    def test_topstep_ruleset_fields(self):
        f = topstep_ruleset(50000)
        assert f.target == 3000
        assert f.drawdown == 2000
        assert f.dll == pytest.approx(1000.0)
        assert f.max_micros == 50       # 5 minis * 10
        assert f.trailing == "eod"
        assert f.min_trading_days == 2
        assert f.consistency == pytest.approx(0.50)
        assert f.lock_level == pytest.approx(0.0)

    def test_topstep_100k_dll(self):
        f = topstep_ruleset(100000)
        assert f.dll == pytest.approx(2000.0)
        assert f.max_micros == 100

    def test_lucid_ruleset_fields(self):
        f = lucid_ruleset(50000)
        assert f.target == 3000
        assert f.drawdown == 2000
        assert f.dll == pytest.approx(600.0)  # 20% of 3000
        assert f.min_trading_days == 5
        assert f.consistency is None
        assert f.static_after_target is True
        assert f.lock_level == pytest.approx(0.0)

    def test_lucid_dll_is_20pct_of_target(self):
        """DLL must be exactly 20 % of target for every Lucid size."""
        for size in [25000, 50000, 100000, 150000]:
            f = lucid_ruleset(size)
            assert f.dll == pytest.approx(0.20 * f.target)

    def test_lucid_25k_initial_floor(self):
        f = lucid_ruleset(25000)
        assert f.initial_floor() == pytest.approx(-1500)  # drawdown=1500

    def test_unknown_apex_size_raises(self):
        with pytest.raises(KeyError):
            apex_ruleset(99999, "intraday")

    def test_unknown_topstep_size_raises(self):
        with pytest.raises(KeyError):
            topstep_ruleset(99999)

    def test_unknown_lucid_size_raises(self):
        with pytest.raises(KeyError):
            lucid_ruleset(99999)


# =============================================================================
# 2. Strategy helpers
# =============================================================================

class TestStrategyHelpers:

    def test_expectancy_scalar_symmetric(self):
        """50% win, 1R/1R => expectancy = 0."""
        s = Strategy(win_rate=0.50, win_R=1.0, loss_R=1.0)
        assert s.expectancy_R() == pytest.approx(0.0)

    def test_expectancy_scalar_positive(self):
        """50% win, 2R win / 1R loss => expectancy = 0.5."""
        s = Strategy(win_rate=0.50, win_R=2.0, loss_R=1.0)
        assert s.expectancy_R() == pytest.approx(0.5)

    def test_expectancy_scalar_negative(self):
        """30% win, 1R win / 2R loss => expectancy = 0.3 - 0.7*2 = -1.1."""
        s = Strategy(win_rate=0.30, win_R=1.0, loss_R=2.0)
        assert s.expectancy_R() == pytest.approx(-1.1)

    def test_profit_factor_2to1(self):
        s = Strategy(win_rate=0.50, win_R=2.0, loss_R=1.0)
        assert s.profit_factor() == pytest.approx(2.0)

    def test_profit_factor_zero_loss_rate(self):
        """100% win rate => profit factor is inf."""
        s = Strategy(win_rate=1.0, win_R=1.0, loss_R=1.0)
        assert math.isinf(s.profit_factor())

    def test_win_R_eff_scalar_passthrough(self):
        s = Strategy(win_rate=0.6, win_R=2.5)
        assert s.win_R_eff() == pytest.approx(2.5)

    def test_loss_R_eff_scalar_passthrough(self):
        s = Strategy(win_rate=0.6, loss_R=0.8)
        assert s.loss_R_eff() == pytest.approx(0.8)

    def test_win_R_eff_mixture_mean(self):
        """Mixture [1.0, 3.0] with equal probs => effective win_R = 2.0."""
        s = Strategy(win_rate=0.6,
                     win_R_mix=[1.0, 3.0], win_R_probs=[0.5, 0.5])
        assert s.win_R_eff() == pytest.approx(2.0)

    def test_loss_R_eff_mixture_mean(self):
        """Mixture [1.0, 0.5] with probs [0.7, 0.3] => effective loss_R = 0.85."""
        s = Strategy(win_rate=0.6,
                     loss_R_mix=[1.0, 0.5], loss_R_probs=[0.7, 0.3])
        assert s.loss_R_eff() == pytest.approx(0.85)

    def test_expectancy_with_mixtures(self):
        """Mixture expectancy uses effective means: 0.6*2 - 0.4*0.85 = 0.86."""
        s = Strategy(win_rate=0.60,
                     win_R_mix=[1.0, 3.0], win_R_probs=[0.5, 0.5],
                     loss_R_mix=[1.0, 0.5], loss_R_probs=[0.7, 0.3])
        assert s.expectancy_R() == pytest.approx(0.86, abs=1e-9)


# =============================================================================
# 3. generate_trade_pnls
# =============================================================================

class TestGenerateTradePnls:

    def test_zero_trades_returns_empty(self):
        strat = Strategy(win_rate=0.6, win_R=1.0, loss_R=1.0, dollar_risk_per_micro=50.0)
        rng = np.random.default_rng(0)
        pnls = generate_trade_pnls(strat, micros=3, n_trades=0, rng=rng)
        assert len(pnls) == 0

    def test_deterministic_with_seed(self):
        """Same seed and parameters produce identical output."""
        strat = Strategy(win_rate=0.60, win_R=2.0, loss_R=1.0,
                         dollar_risk_per_micro=50.0,
                         cost_per_trade_per_micro=2.0)
        pnls_a = generate_trade_pnls(strat, micros=3, n_trades=10, rng=np.random.default_rng(42))
        pnls_b = generate_trade_pnls(strat, micros=3, n_trades=10, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(pnls_a, pnls_b)

    def test_cost_deducted_on_wins(self):
        """100% win rate: each trade = win_R * risk - cost * micros."""
        strat = Strategy(win_rate=1.0, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=50.0,
                         cost_per_trade_per_micro=2.0)
        rng = np.random.default_rng(0)
        pnls = generate_trade_pnls(strat, micros=3, n_trades=5, rng=rng)
        # expected: 1.0 * 50 * 3 - 2.0 * 3 = 144 per trade
        np.testing.assert_allclose(pnls, np.full(5, 144.0))

    def test_cost_deducted_on_losses(self):
        """0% win rate: each trade = -loss_R * risk - cost * micros."""
        strat = Strategy(win_rate=0.0, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=50.0,
                         cost_per_trade_per_micro=2.0)
        rng = np.random.default_rng(0)
        pnls = generate_trade_pnls(strat, micros=3, n_trades=5, rng=rng)
        # expected: -1.0 * 50 * 3 - 2.0 * 3 = -156 per trade
        np.testing.assert_allclose(pnls, np.full(5, -156.0))

    def test_r_sigma_adds_variation(self):
        """With r_sigma > 0, outcomes are not all identical even for 100% win rate."""
        strat = Strategy(win_rate=1.0, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=100.0,
                         r_sigma=0.5)
        rng = np.random.default_rng(5)
        pnls = generate_trade_pnls(strat, micros=1, n_trades=10, rng=rng)
        # Should not all be the same value
        assert len(set(pnls.tolist())) > 1

    def test_cluster_rho_produces_sticky_outcomes(self):
        """cluster_rho near 1 creates very long runs (few sign changes)."""
        strat = Strategy(win_rate=0.50, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=10.0,
                         cluster_rho=0.99)
        rng = np.random.default_rng(100)
        pnls = generate_trade_pnls(strat, micros=1, n_trades=50, rng=rng)
        # Count sign-change events; with rho=0.99 over 50 trades there should be very few
        signs = np.sign(pnls)
        transitions = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        assert transitions < 10, f"Expected sticky outcomes but got {transitions} transitions"

    def test_iid_cluster_rho_zero(self):
        """cluster_rho=0 should produce i.i.d. outcomes (no systematic stickiness)."""
        strat = Strategy(win_rate=0.50, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=10.0,
                         cluster_rho=0.0)
        rng = np.random.default_rng(77)
        pnls = generate_trade_pnls(strat, micros=1, n_trades=200, rng=rng)
        signs = np.sign(pnls)
        transitions = sum(1 for i in range(1, len(signs)) if signs[i] != signs[i - 1])
        # With i.i.d. 50/50 outcomes over 200 trades, expect ~100 transitions; far more than sticky
        assert transitions > 20, f"Expected many transitions for i.i.d. but got {transitions}"

    def test_win_R_mixture_used(self):
        """Wins drawn from a mixture should match the mixture's mean over many draws."""
        strat = Strategy(win_rate=1.0,
                         win_R_mix=[1.0, 3.0], win_R_probs=[0.5, 0.5],
                         dollar_risk_per_micro=10.0)
        rng = np.random.default_rng(0)
        pnls = generate_trade_pnls(strat, micros=1, n_trades=10000, rng=rng)
        mean_R = pnls.mean() / 10.0  # divide by risk_dollars (10 * 1 micro)
        assert mean_R == pytest.approx(2.0, abs=0.05)  # mixture mean = 2.0


# =============================================================================
# 4. _assign_trades_to_days
# =============================================================================

class TestAssignTradesToDays:

    def test_zero_trades_returns_all_zeros(self):
        rng = np.random.default_rng(0)
        counts = _assign_trades_to_days(0, 5, rng)
        assert counts == [0, 0, 0, 0, 0]

    def test_sum_equals_n_trades(self):
        rng = np.random.default_rng(42)
        counts = _assign_trades_to_days(100, 21, rng)
        assert sum(counts) == 100
        assert len(counts) == 21

    def test_negative_trades_treated_as_zero(self):
        """n_trades <= 0 should return all zeros (guarded in the docstring)."""
        rng = np.random.default_rng(0)
        counts = _assign_trades_to_days(-5, 3, rng)
        assert counts == [0, 0, 0]

    def test_single_day(self):
        rng = np.random.default_rng(1)
        counts = _assign_trades_to_days(7, 1, rng)
        assert counts == [7]

    def test_all_nonnegative(self):
        rng = np.random.default_rng(99)
        counts = _assign_trades_to_days(50, 10, rng)
        assert all(c >= 0 for c in counts)


# =============================================================================
# 5. simulate_eval — aggregate stats with fixed seeds
# =============================================================================

class TestSimulateEvalAggregateStats:

    def test_huge_edge_passes_always(self):
        """90% win, +1R/-0.25R at small risk should pass ~100% of the time."""
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.90, win_R=1.0, loss_R=0.25,
                         dollar_risk_per_micro=20.0, trades_per_month=120)
        r = simulate_eval(firm, strat, micros=4, n_sims=1000, seed=42)
        assert r["pass_rate"] == pytest.approx(1.0)
        assert r["fail_hit_trailing_floor"] == pytest.approx(0.0)
        assert r["fail_timeout_no_target"] == pytest.approx(0.0)

    def test_negative_edge_fails_always(self):
        """30% win, +1R/-2R is strongly negative EV — should never pass."""
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.30, win_R=1.0, loss_R=2.0,
                         dollar_risk_per_micro=100.0, trades_per_month=100)
        r = simulate_eval(firm, strat, micros=4, n_sims=1000, seed=42)
        assert r["pass_rate"] == pytest.approx(0.0)
        assert r["fail_hit_trailing_floor"] == pytest.approx(1.0)

    def test_zero_edge_below_barrier_ceiling(self):
        """50% win, +1R/-1R: pass rate must stay below the 40% driftless barrier ceiling
        and the trailing floor must be the dominant fail mode."""
        strat = Strategy(win_rate=0.50, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=100.0, trades_per_month=150)
        r = simulate_eval(apex_ruleset(50000, "intraday"), strat, micros=4,
                          n_sims=2000, seed=42)
        assert r["pass_rate"] < 0.40, "Zero-edge pass rate must be below the 40% barrier ceiling"
        assert r["fail_hit_trailing_floor"] > r["fail_timeout_no_target"], \
            "Trailing breach must be the dominant fail mode for a zero-edge walk"

    def test_fraction_sum_equals_one(self):
        """All outcome fractions must sum to exactly 1.0."""
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r = simulate_eval(firm, strat, micros=4, n_sims=500, seed=99)
        total = (r["pass_rate"] + r["fail_hit_trailing_floor"] +
                 r["fail_timeout_no_target"] + r["fail_consistency"])
        assert total == pytest.approx(1.0)

    def test_median_days_nan_when_no_pass(self):
        """When no simulation passes, median and mean days should be nan."""
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.20, win_R=1.0, loss_R=2.0,
                         dollar_risk_per_micro=200.0, trades_per_month=50)
        r = simulate_eval(firm, strat, micros=4, n_sims=200, seed=1)
        assert r["pass_rate"] == pytest.approx(0.0)
        assert math.isnan(r["median_days_to_pass"])
        assert math.isnan(r["mean_days_to_pass"])

    def test_strategy_expectancy_in_result(self):
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0)
        r = simulate_eval(apex_ruleset(50000, "intraday"), strat, micros=4,
                          n_sims=100, seed=1)
        assert r["strategy_expectancy_R"] == pytest.approx(strat.expectancy_R())

    def test_result_n_sims_matches_input(self):
        r = simulate_eval(apex_ruleset(50000, "intraday"),
                          Strategy(win_rate=0.6, win_R=1.5, loss_R=1.0,
                                   dollar_risk_per_micro=20.0),
                          micros=2, n_sims=137, seed=0)
        assert r["n_sims"] == 137

    def test_n_sims_1_runs_without_error(self):
        r = simulate_eval(apex_ruleset(50000, "intraday"),
                          Strategy(win_rate=0.60, win_R=1.5, loss_R=1.0,
                                   dollar_risk_per_micro=20.0),
                          micros=2, n_sims=1, seed=5)
        total = (r["pass_rate"] + r["fail_hit_trailing_floor"] +
                 r["fail_timeout_no_target"] + r["fail_consistency"])
        assert total == pytest.approx(1.0)

    def test_deterministic_results_with_seed(self):
        """Running simulate_eval twice with the same seed returns identical results."""
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r1 = simulate_eval(firm, strat, micros=4, n_sims=500, seed=99)
        r2 = simulate_eval(firm, strat, micros=4, n_sims=500, seed=99)
        assert r1["pass_rate"] == pytest.approx(r2["pass_rate"])
        assert r1["fail_hit_trailing_floor"] == pytest.approx(r2["fail_hit_trailing_floor"])

    def test_topstep_fixed_seed_snapshot(self):
        """Regression snapshot: TopStep 50k with seed=7, n_sims=1000."""
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r = simulate_eval(topstep_ruleset(50000), strat, micros=5, n_sims=1000, seed=7)
        assert r["pass_rate"] == pytest.approx(0.834, abs=1e-9)
        assert r["trailing"] == "eod"
        assert r["micros"] == 5

    def test_lucid_fixed_seed_snapshot(self):
        """Regression snapshot: Lucid 50k with seed=7, n_sims=1000."""
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r = simulate_eval(lucid_ruleset(50000), strat, micros=5, n_sims=1000, seed=7)
        assert r["pass_rate"] == pytest.approx(0.829, abs=1e-9)
        assert r["trailing"] == "eod"


# =============================================================================
# 6. Micros clamping
# =============================================================================

class TestMicrosClamping:

    def test_micros_above_max_clamped(self):
        """Requesting more micros than the firm cap must be silently clamped."""
        firm = apex_ruleset(50000, "intraday")  # max_micros = 60
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r = simulate_eval(firm, strat, micros=999, n_sims=10, seed=1)
        assert r["micros"] == 60

    def test_micros_zero_clamped_to_one(self):
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0)
        r = simulate_eval(firm, strat, micros=0, n_sims=10, seed=1)
        assert r["micros"] == 1

    def test_micros_negative_clamped_to_one(self):
        firm = apex_ruleset(50000, "intraday")
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0)
        r = simulate_eval(firm, strat, micros=-5, n_sims=10, seed=1)
        assert r["micros"] == 1

    def test_micros_at_max_not_clamped(self):
        firm = apex_ruleset(50000, "intraday")  # max_micros = 60
        strat = Strategy(win_rate=0.55, win_R=1.5, loss_R=1.0,
                         dollar_risk_per_micro=30.0, trades_per_month=80)
        r = simulate_eval(firm, strat, micros=60, n_sims=10, seed=1)
        assert r["micros"] == 60


# =============================================================================
# 7. Trailing mechanics: intraday vs EOD
# =============================================================================

class TestTrailingMechanics:

    def _patch_and_run(self, firm, day_scripts, trade_counts_per_day):
        """Run _run_one_eval with a scripted P&L sequence and fixed day trade counts.

        day_scripts : list of np.ndarray, one per trading day that has trades
        trade_counts_per_day : list of int, length = trading_days_in_horizon

        Patches generate_trade_pnls and _assign_trades_to_days on the module,
        runs the eval, then restores the originals.
        """
        saved_gen = e.generate_trade_pnls
        saved_assign = e._assign_trades_to_days
        call_state = {"idx": 0}

        def fake_gen(strategy, micros, n_trades, rng):
            if n_trades == 0:
                return np.zeros(0)
            out = day_scripts[call_state["idx"]]
            call_state["idx"] += 1
            return out

        def fake_assign(n_trades, n_days, rng):
            return list(trade_counts_per_day)

        e.generate_trade_pnls = fake_gen
        e._assign_trades_to_days = fake_assign
        dummy = Strategy(win_rate=0.5, trades_per_month=sum(trade_counts_per_day),
                         dollar_risk_per_micro=1.0)
        try:
            result, days = _run_one_eval(firm, dummy, micros=1,
                                         rng=np.random.default_rng(0))
        finally:
            e.generate_trade_pnls = saved_gen
            e._assign_trades_to_days = saved_assign
        return result, days

    def test_intraday_ratchet_kills_spike_giveback(self):
        """Apex 50k intraday: spike to +800 then give back to +100 on day 0,
        followed by a -1300 trade on day 1.

        The intraday floor ratchets to min(800 - 2000, 100) = -1200 on the +800 spike.
        After day-0 close at +100 the floor stays at -1200 (still tracks the +800 peak).
        Day-1 trade: equity = 100 - 1300 = -1200 <= -1200 => FAIL_TRAILING.
        """
        firm = apex_ruleset(50000, "intraday")
        n_days = firm.trading_days_in_horizon

        # Day 0: two trades [+800, -700]; Day 1: one trade [-1300]; rest: 0 trades
        scripts = [np.array([800.0, -700.0]), np.array([-1300.0])]
        trade_counts = [2, 1] + [0] * (n_days - 2)

        result, _ = self._patch_and_run(firm, scripts, trade_counts)
        assert result == FAIL_TRAILING, \
            "Intraday ratchet should fail the account on a spike-then-give-back path"

    def test_eod_survives_same_spike_giveback(self):
        """Apex 50k EOD: same path as above.

        EOD floor only updates at day close. Day-0 closes at +100 (eod_peak=100),
        so day-1's floor = min(100 - 2000, 100) = -1900.
        Day-1 trade: equity = 100 - 1300 = -1200 > -1900 => SURVIVES (not FAIL_TRAILING).
        """
        firm = apex_ruleset(50000, "eod")
        n_days = firm.trading_days_in_horizon

        scripts = [np.array([800.0, -700.0]), np.array([-1300.0])]
        trade_counts = [2, 1] + [0] * (n_days - 2)

        result, _ = self._patch_and_run(firm, scripts, trade_counts)
        assert result != FAIL_TRAILING, \
            "EOD trailing should not fail the account on the same spike-then-give-back path"

    def test_intraday_stricter_than_eod_aggregate(self):
        """For a swingy strategy, intraday pass rate must be strictly lower than EOD."""
        strat = Strategy(win_rate=0.50, win_R=2.2, loss_R=1.0,
                         dollar_risk_per_micro=50.0, trades_per_month=140,
                         r_sigma=0.40)
        r_intra = simulate_eval(apex_ruleset(50000, "intraday"), strat,
                                micros=6, n_sims=5000, seed=999)
        r_eod = simulate_eval(apex_ruleset(50000, "eod"), strat,
                              micros=6, n_sims=5000, seed=999)
        assert r_intra["pass_rate"] < r_eod["pass_rate"], \
            "Intraday trailing must produce a lower pass rate than EOD for a swingy strategy"


# =============================================================================
# 8. Lock level mechanics
# =============================================================================

class TestLockLevel:

    def _run_scripted(self, firm, scripts, per_day_counts):
        """Thin wrapper around the patch helper above."""
        helper = TestTrailingMechanics()
        return helper._patch_and_run(firm, scripts, per_day_counts)

    def test_lock_at_zero_survivor(self):
        """EOD trailing with lock_level=0 (TopStep/Lucid style):
        Day 0 closes +2500. EOD peak=2500; floor = min(2500-2000, 0) = 0 (locked at 0).
        Day 2: a -2450 trade => equity = 2500 + 0 + (-2450) = 50 > floor 0 => SURVIVES.
        """
        firm = apex_ruleset(50000, "eod")
        firm.lock_level = 0.0
        n = firm.trading_days_in_horizon

        scripts = [np.array([2500.0]), np.array([0.0]), np.array([-2450.0])]
        counts = [1, 1, 1] + [0] * (n - 3)
        result, _ = self._run_scripted(firm, scripts, counts)
        assert result != FAIL_TRAILING, \
            "With lock at 0, a +2500 close should freeze the floor at 0 and let +50 survive"

    def test_lock_at_zero_failure(self):
        """Same setup but the final drop is -2550 => equity = -50 < floor 0 => FAIL."""
        firm = apex_ruleset(50000, "eod")
        firm.lock_level = 0.0
        n = firm.trading_days_in_horizon

        scripts = [np.array([2500.0]), np.array([0.0]), np.array([-2550.0])]
        counts = [1, 1, 1] + [0] * (n - 3)
        result, _ = self._run_scripted(firm, scripts, counts)
        assert result == FAIL_TRAILING, \
            "With lock at 0, a -2550 drop from +2500 should breach the locked floor at 0"

    def test_no_lock_control_case(self):
        """Without a lock (lock_level=1e9), EOD peak=2500 => floor = 500.
        A -2450 trade from +2500 gives equity = +50, which is still > floor=500? No:
        +50 < 500 => FAIL. This proves the lock is load-bearing: it is the ONLY reason
        the survivor case above passes.
        """
        firm = apex_ruleset(50000, "eod")
        firm.lock_level = 1e9   # effectively no lock
        n = firm.trading_days_in_horizon

        scripts = [np.array([2500.0]), np.array([0.0]), np.array([-2450.0])]
        counts = [1, 1, 1] + [0] * (n - 3)
        result, _ = self._run_scripted(firm, scripts, counts)
        assert result == FAIL_TRAILING, \
            "Without a lock, floor=500 after +2500 close; +50 should breach it"


# =============================================================================
# 9. Daily Loss Limit (DLL)
# =============================================================================

class TestDailyLossLimit:

    def _run_scripted(self, firm, scripts, per_day_counts):
        helper = TestTrailingMechanics()
        return helper._patch_and_run(firm, scripts, per_day_counts)

    def test_dll_caps_day_and_eval_survives(self):
        """TopStep 50k DLL=1000. A single -1500 trade should be capped at -1000.
        Starting floor = -2000. After cap: equity = -1000 > -2000 => eval survives.
        With no more trades the eval ends as FAIL_TIMEOUT (not FAIL_TRAILING).
        """
        firm = topstep_ruleset(50000)
        n = firm.trading_days_in_horizon

        scripts = [np.array([-1500.0])]
        counts = [1] + [0] * (n - 1)
        result, _ = self._run_scripted(firm, scripts, counts)
        assert result == FAIL_TIMEOUT, \
            "DLL should cap the day's loss and leave the eval alive to time out"

    def test_dll_breach_plus_floor_fails(self):
        """TopStep 50k DLL=1000. A -2500 trade without DLL would blow -2000 floor.
        WITH DLL it caps at -1000, so equity=-1000 > floor=-2000 => SURVIVES.
        The same -2500 trade on Apex 50k (no DLL, same drawdown) => FAIL_TRAILING.
        """
        firm_dll = topstep_ruleset(50000)      # DLL=1000, drawdown=2000
        firm_nodll = apex_ruleset(50000, "eod")  # no DLL, drawdown=2000
        n = firm_dll.trading_days_in_horizon

        scripts = [np.array([-2500.0])]
        counts = [1] + [0] * (n - 1)

        r_dll, _ = self._run_scripted(firm_dll, scripts, counts)
        r_nodll, _ = self._run_scripted(firm_nodll, scripts, counts)

        assert r_nodll == FAIL_TRAILING, "Without DLL a -2500 trade should breach the -2000 floor"
        assert r_dll == FAIL_TIMEOUT, "With DLL the -2500 trade is capped at -1000; eval should time out"

    def test_dll_aggregate_positive_effect(self):
        """With a strategy that routinely breaches the DLL, having a DLL should produce
        a higher pass rate than no DLL (or at least not lower) because it prevents the worst
        intraday moves from reaching the trailing floor on big-loss days.

        We compare TopStep 50k (DLL=1000) vs a custom firm with the same rules but no DLL.
        """
        strat = Strategy(win_rate=0.45, win_R=2.0, loss_R=1.5,
                         dollar_risk_per_micro=50.0, trades_per_month=80)
        firm_with_dll = topstep_ruleset(50000)

        firm_no_dll = FirmRuleset(
            name="NODLL_50k", size=50000, target=3000, drawdown=2000,
            trailing="eod", max_micros=50, dll=None, min_trading_days=0,
            consistency=None, lock_level=0.0,
        )

        r_dll = simulate_eval(firm_with_dll, strat, micros=5, n_sims=2000, seed=55)
        r_no = simulate_eval(firm_no_dll, strat, micros=5, n_sims=2000, seed=55)

        # DLL prevents floor breaches; pass rate should not be worse
        assert r_dll["fail_hit_trailing_floor"] <= r_no["fail_hit_trailing_floor"] + 0.05, \
            "DLL should reduce or not increase trailing-floor failures"


# =============================================================================
# 10. Minimum trading days (Lucid)
# =============================================================================

class TestMinTradingDays:

    def test_lucid_min_days_forces_longer_pass(self):
        """A huge edge on Apex (min_days=0) passes quickly (median day 1).
        The same edge on Lucid (min_days=5) must have median_days_to_pass >= 5.
        """
        strat = Strategy(win_rate=0.99, win_R=1.0, loss_R=0.1,
                         dollar_risk_per_micro=60.0, trades_per_month=200)
        r_apex = simulate_eval(apex_ruleset(50000, "intraday"), strat,
                               micros=6, n_sims=2000, seed=3)
        r_lucid = simulate_eval(lucid_ruleset(50000), strat,
                                micros=6, n_sims=2000, seed=3)

        assert r_lucid["median_days_to_pass"] >= 5, \
            "Lucid min 5 trading days must show up in median_days_to_pass"
        assert r_apex["median_days_to_pass"] < r_lucid["median_days_to_pass"], \
            "Apex (no min-days constraint) must finish faster than Lucid"

    def test_apex_no_min_days_passes_day_one(self):
        """With a huge edge Apex should be able to pass on day 1 (median_days=1)."""
        strat = Strategy(win_rate=0.99, win_R=1.0, loss_R=0.1,
                         dollar_risk_per_micro=60.0, trades_per_month=200)
        r = simulate_eval(apex_ruleset(50000, "intraday"), strat,
                          micros=6, n_sims=2000, seed=3)
        assert r["median_days_to_pass"] == pytest.approx(1.0), \
            "Huge-edge Apex should pass on day 1 (no min-days gate)"


# =============================================================================
# 11. Consistency rule (TopStep)
# =============================================================================

class TestConsistencyRule:

    def test_lumpy_trader_gets_consistency_failures(self):
        """A lumpy strategy (few large trades per month) that tends to hit the target
        in a single huge day should show non-zero consistency_fail rate on TopStep.
        """
        strat = Strategy(win_rate=0.80, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=300.0, trades_per_month=12)
        r = simulate_eval(topstep_ruleset(50000), strat,
                          micros=6, n_sims=6000, seed=5)
        assert r["fail_consistency"] > 0.0, \
            "Lumpy strategy must trigger TopStep consistency failures"

    def test_apex_no_consistency_failures(self):
        """Apex has no consistency rule; consistency_fail should always be 0."""
        strat = Strategy(win_rate=0.80, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=300.0, trades_per_month=12)
        r = simulate_eval(apex_ruleset(50000, "intraday"), strat,
                          micros=6, n_sims=2000, seed=5)
        assert r["fail_consistency"] == pytest.approx(0.0)

    def test_lucid_no_consistency_failures(self):
        """Lucid has no consistency rule."""
        strat = Strategy(win_rate=0.80, win_R=1.0, loss_R=1.0,
                         dollar_risk_per_micro=300.0, trades_per_month=12)
        r = simulate_eval(lucid_ruleset(50000), strat,
                          micros=6, n_sims=2000, seed=5)
        assert r["fail_consistency"] == pytest.approx(0.0)


# =============================================================================
# 12. Contract-size hump
# =============================================================================

class TestContractSizeHump:

    def test_pass_rate_hump_over_micros(self):
        """As micros increases from 1 to 60 (Apex 50k cap), the pass rate should
        first rise (more $/trade => reach target faster) then fall (more $/trade =>
        risk blowing the trailing floor). The peak must occur at an interior size.
        """
        strat = Strategy(win_rate=0.53, win_R=1.3, loss_R=1.0,
                         dollar_risk_per_micro=25.0, trades_per_month=90,
                         r_sigma=0.10)
        firm = apex_ruleset(50000, "intraday")
        sweep = [1, 2, 4, 8, 16, 30, 60]
        passes = [
            simulate_eval(firm, strat, micros=m, n_sims=3000, seed=77)["pass_rate"]
            for m in sweep
        ]
        peak_idx = int(np.argmax(passes))

        rose = passes[peak_idx] > passes[0] + 1e-6
        fell = passes[-1] < passes[peak_idx] - 1e-6
        interior = 0 < peak_idx < len(passes) - 1

        assert rose, f"Pass rate must rise from the smallest size to the peak (got {passes})"
        assert fell, f"Pass rate must fall from peak to the largest size (got {passes})"
        assert interior, f"Peak must be at an interior size, not first or last (peak_idx={peak_idx})"


# =============================================================================
# 13. Built-in self-tests pass via run_self_tests()
# =============================================================================

class TestBuiltinSelfTests:

    def test_self_tests_all_pass(self):
        """The module ships with its own self-test suite; it must pass green."""
        from eval_sim import run_self_tests
        all_passed, detail = run_self_tests(n_sims=10000, seed=12345)
        assert all_passed, f"Built-in self-tests failed:\n{detail}"
