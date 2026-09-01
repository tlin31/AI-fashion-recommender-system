"""Unit tests for the regression gate.

The cases that matter are the ones where a naive gate returns the wrong verdict
confidently: an improvement in a lower-is-better metric, a comparison across two
different cohorts, and a degraded run being promoted to baseline.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from check_regression import check, lock  # noqa: E402

PASS, FAIL, REFUSED = 0, 1, 2


def payload(*, ndcg=0.50, gini=0.60, users=100, catalogue=95335,
            degraded=False, arm="GorseItemToItem", group="warm"):
    return {
        "generated_at": "2026-09-01T00:00:00+00:00",
        "degraded": degraded,
        "preconditions": {"problems": ["evicted"] if degraded else []},
        "cohort": {"evaluable_users": users, "by_cohort": {"warm": users},
                   "warm_strata": {"1": users}},
        "denominators": {"catalogue_size": catalogue},
        "arms": [{"arm": arm, "groups": {group: {
            "ndcg@10": ndcg, "recall@10": 0.30, "recall@20": 0.40,
            "hit_rate@10": 0.35, "mrr": 0.25,
            "catalog_coverage": 0.10, "gini": gini, "novelty": 8.0, "ild": 0.60,
            "recall@10_ceiling": 1.0, "n_users": users, "catalog_size": catalogue,
        }}}],
    }


def write(tmp_path, name, data):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ── Direction ────────────────────────────────────────────────────────────────

def test_accuracy_drop_beyond_threshold_fails(tmp_path, capsys):
    b = write(tmp_path, "b.json", payload(ndcg=0.50))
    l = write(tmp_path, "l.json", payload(ndcg=0.40))     # -20%
    assert check(0.05, l, b) == FAIL
    assert "ndcg@10" in capsys.readouterr().out


def test_accuracy_drop_within_threshold_passes(tmp_path):
    b = write(tmp_path, "b.json", payload(ndcg=0.50))
    l = write(tmp_path, "l.json", payload(ndcg=0.49))     # -2%
    assert check(0.05, l, b) == PASS


def test_gini_falling_is_an_improvement_not_a_regression(tmp_path, capsys):
    """Gini is better when lower. A gate that only knows 'value dropped' would
    report every improvement in exposure concentration as a failure."""
    b = write(tmp_path, "b.json", payload(gini=0.80))
    l = write(tmp_path, "l.json", payload(gini=0.40))     # halved = much better
    assert check(0.05, l, b) == PASS
    assert "better" in capsys.readouterr().out


# ── Gated vs reported ────────────────────────────────────────────────────────

def test_beyond_accuracy_movement_is_reported_not_failed(tmp_path, capsys):
    """Diversity trades against accuracy; gating it would forbid the trade-off
    rather than surface it."""
    b = write(tmp_path, "b.json", payload())
    l_data = payload()
    l_data["arms"][0]["groups"]["warm"]["ild"] = 0.10      # collapsed
    l = write(tmp_path, "l.json", l_data)
    assert check(0.05, l, b) == PASS
    out = capsys.readouterr().out
    assert "ild" in out and "movements" in out


def test_ceiling_and_cohort_size_are_never_compared(tmp_path):
    b = write(tmp_path, "b.json", payload())
    l_data = payload()
    l_data["arms"][0]["groups"]["warm"]["recall@10_ceiling"] = 0.1   # collapsed
    l = write(tmp_path, "l.json", l_data)
    assert check(0.05, l, b) == PASS


# ── Comparability ────────────────────────────────────────────────────────────

def test_different_cohort_is_inconclusive_not_a_pass(tmp_path, capsys):
    """The cohort is sampled. Metrics over different populations are not
    comparable in either direction -- returning 'pass' would be a false clean
    bill of health."""
    b = write(tmp_path, "b.json", payload(users=100, ndcg=0.50))
    l = write(tmp_path, "l.json", payload(users=5000, ndcg=0.99))
    assert check(0.05, l, b) == FAIL
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_changed_catalogue_is_inconclusive(tmp_path, capsys):
    b = write(tmp_path, "b.json", payload(catalogue=95335))
    l = write(tmp_path, "l.json", payload(catalogue=5000))
    assert check(0.05, l, b) == FAIL
    assert "catalogue changed" in capsys.readouterr().out


def test_degraded_latest_is_inconclusive(tmp_path, capsys):
    b = write(tmp_path, "b.json", payload())
    l = write(tmp_path, "l.json", payload(degraded=True))
    assert check(0.05, l, b) == FAIL
    assert "degraded" in capsys.readouterr().out


def test_no_shared_arms_is_inconclusive(tmp_path, capsys):
    b = write(tmp_path, "b.json", payload(arm="GorseCF"))
    l = write(tmp_path, "l.json", payload(arm="Random"))
    assert check(0.05, l, b) == FAIL
    assert "INCONCLUSIVE" in capsys.readouterr().out


def test_new_arm_is_noted_but_does_not_fail(tmp_path, capsys):
    b = write(tmp_path, "b.json", payload())
    l_data = payload()
    l_data["arms"].append({"arm": "BrandNew", "groups": {"warm": {"ndcg@10": 0.01}}})
    l = write(tmp_path, "l.json", l_data)
    assert check(0.05, l, b) == PASS
    assert "BrandNew" in capsys.readouterr().out


# ── Locking ──────────────────────────────────────────────────────────────────

def test_lock_refuses_a_degraded_run(tmp_path, capsys):
    """A baseline is what every later result is measured against; locking one
    whose preconditions failed poisons them all, undetectably."""
    l = write(tmp_path, "l.json", payload(degraded=True))
    assert lock(l, tmp_path / "b.json", force=False) == REFUSED
    assert not (tmp_path / "b.json").exists()


def test_lock_writes_baseline_with_a_locked_at_stamp(tmp_path):
    l = write(tmp_path, "l.json", payload())
    b = tmp_path / "b.json"
    assert lock(l, b, force=False) == PASS
    assert json.loads(b.read_text())["locked_at"] == "2026-09-01T00:00:00+00:00"


def test_lock_force_overrides_but_keeps_the_degraded_flag(tmp_path):
    l = write(tmp_path, "l.json", payload(degraded=True))
    b = tmp_path / "b.json"
    assert lock(l, b, force=True) == PASS
    assert json.loads(b.read_text())["degraded"] is True


# ── --incomparable: a metric whose definition changed this round ─────────────

def _payload(ild: float, coverage: float = 0.30, ndcg: float = 0.02) -> dict:
    return {"arms": [{"arm": "A", "groups": {"all": {
        "ndcg@10": ndcg, "ild": ild, "catalog_coverage": coverage,
    }}}]}


def test_incomparable_metric_is_reported_without_a_verdict(tmp_path, capsys):
    base = tmp_path / "b.json"; latest = tmp_path / "l.json"
    base.write_text(json.dumps(_payload(ild=0.44, coverage=0.30)))
    latest.write_text(json.dumps(_payload(ild=0.71, coverage=0.42)))

    rc = check(0.05, latest, base, incomparable=("ild",))
    out = capsys.readouterr().out
    assert rc == 0
    assert "Not comparable this round" in out
    assert "0.4400 -> 0.7100" in out
    # The whole point: no direction is asserted for a metric that changed
    # definition, while a genuinely comparable one still gets one.
    assert "better" not in out.split("Not comparable")[1]
    assert "catalog_coverage" in out.split("Beyond-accuracy")[1]


def test_comparable_metrics_still_get_a_direction(tmp_path, capsys):
    base = tmp_path / "b.json"; latest = tmp_path / "l.json"
    base.write_text(json.dumps(_payload(ild=0.44)))
    latest.write_text(json.dumps(_payload(ild=0.71)))

    check(0.05, latest, base)
    out = capsys.readouterr().out
    assert "Not comparable this round" not in out
    assert "ild" in out and ("better" in out or "worse" in out)
