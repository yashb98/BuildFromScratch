"""Tests for research/ablation_config.py — the validated multi-seed, per-arm-tuned
cohort builder that makes the next ablation a 'clean win'. Pure, deterministic,
stdlib-only."""
import pytest

import ablation_config as ac
from data_decontam import cell_split_seed


def _good_arms():
    return [
        ac.Arm("baseline", hparams={"lr": 3e-4, "wd": 0.1}, is_baseline=True),
        ac.Arm("treatment", hparams={"lr": 4e-4, "wd": 0.05}),  # PER-ARM tuned
    ]


# --- Arm validation ----------------------------------------------------------

def test_arm_requires_per_arm_hparams():
    with pytest.raises(ValueError):
        ac.Arm("t", hparams={}, is_baseline=False).validate()  # empty == shared confound


def test_arm_rejects_non_finite_hparam():
    with pytest.raises(ValueError):
        ac.Arm("t", hparams={"lr": float("nan")}).validate()
    with pytest.raises(ValueError):
        ac.Arm("t", hparams={"lr": float("inf")}).validate()


def test_arm_accepts_numbers_bools_strings():
    ac.Arm("t", hparams={"lr": 3e-4, "use_muon": True, "sched": "cosine"}).validate()


# --- n_seeds gate ------------------------------------------------------------

def test_validate_refuses_fewer_than_three_seeds():
    with pytest.raises(ValueError):
        ac.build_cohort(_good_arms(), n_seeds=2)
    with pytest.raises(ValueError):
        ac.build_cohort(_good_arms(), n_seeds=1)


def test_three_seeds_builds_but_is_scoped_not_clean():
    cfg = ac.build_cohort(_good_arms(), n_seeds=3, randomized_split=True)
    assert cfg.verdict_tier() == "scoped"   # >=3 builds, but <5 -> not clean
    assert cfg.is_clean_tier is False


def test_clean_tier_requires_five_seeds_and_randomized_split():
    clean = ac.build_cohort(_good_arms(), n_seeds=5, randomized_split=True)
    assert clean.is_clean_tier is True and clean.verdict_tier() == "clean"
    # 5 seeds but FIXED split -> only scoped (init+shuffle variance only)
    fixed = ac.build_cohort(_good_arms(), n_seeds=5, randomized_split=False)
    assert fixed.is_clean_tier is False and fixed.verdict_tier() == "scoped"


# --- arm/baseline structural validation -------------------------------------

def test_requires_exactly_one_baseline():
    two_base = [ac.Arm("a", {"lr": 1e-4}, is_baseline=True),
                ac.Arm("b", {"lr": 1e-4}, is_baseline=True)]
    with pytest.raises(ValueError):
        ac.build_cohort(two_base, n_seeds=5)
    no_base = [ac.Arm("a", {"lr": 1e-4}), ac.Arm("b", {"lr": 1e-4})]
    with pytest.raises(ValueError):
        ac.build_cohort(no_base, n_seeds=5)


def test_requires_at_least_two_arms():
    with pytest.raises(ValueError):
        ac.build_cohort([ac.Arm("solo", {"lr": 1e-4}, is_baseline=True)], n_seeds=5)


def test_rejects_duplicate_arm_names():
    arms = [ac.Arm("x", {"lr": 1e-4}, is_baseline=True), ac.Arm("x", {"lr": 2e-4})]
    with pytest.raises(ValueError):
        ac.build_cohort(arms, n_seeds=5)


def test_rejects_bad_val_fraction_and_direction():
    with pytest.raises(ValueError):
        ac.build_cohort(_good_arms(), n_seeds=5, val_fraction=0.0)
    with pytest.raises(ValueError):
        ac.build_cohort(_good_arms(), n_seeds=5, direction="bigger")


# --- per-arm hparams flow through to cells ----------------------------------

def test_cells_carry_per_arm_hparams():
    cfg = ac.build_cohort(_good_arms(), n_seeds=5)
    cells = cfg.cells()
    assert len(cells) == 2 * 5  # arms x seeds
    base_cells = [c for c in cells if c["arm"] == "baseline"]
    treat_cells = [c for c in cells if c["arm"] == "treatment"]
    assert all(c["hparams"] == {"lr": 3e-4, "wd": 0.1} for c in base_cells)
    assert all(c["hparams"] == {"lr": 4e-4, "wd": 0.05} for c in treat_cells)


# --- split-seed wiring matches data_decontam --------------------------------

def test_fixed_split_every_cell_same_split_seed():
    cfg = ac.build_cohort(_good_arms(), n_seeds=5, base_split_seed=11,
                          randomized_split=False)
    seeds = {c["split_seed"] for c in cfg.cells()}
    assert seeds == {11}  # fixed: one split for the whole cohort


def test_randomized_split_distinct_split_seed_per_train_seed():
    cfg = ac.build_cohort(_good_arms(), n_seeds=5, base_split_seed=11,
                          randomized_split=True)
    cells = cfg.cells()
    # the split seed each cell uses must match data_decontam's derivation exactly
    for c in cells:
        assert c["split_seed"] == cell_split_seed(11, c["train_seed"], randomized=True)
    # across the 5 distinct train seeds there are 5 distinct split seeds
    per_seed = {c["train_seed"]: c["split_seed"] for c in cells}
    assert len(set(per_seed.values())) == 5
    # both arms at the SAME train seed share the SAME split (paired comparison)
    by_seed = {}
    for c in cells:
        by_seed.setdefault(c["train_seed"], set()).add(c["split_seed"])
    assert all(len(v) == 1 for v in by_seed.values())


def test_seed_start_shifts_the_seed_grid():
    cfg = ac.build_cohort(_good_arms(), n_seeds=5, seed_start=100)
    assert cfg.seeds() == [100, 101, 102, 103, 104]


# --- serialization / dict-arm convenience -----------------------------------

def test_build_from_dict_arms_and_to_dict():
    arms = [{"name": "base", "hparams": {"lr": 3e-4}, "is_baseline": True},
            {"name": "treat", "hparams": {"lr": 4e-4}}]
    cfg = ac.build_cohort(arms, n_seeds=5, randomized_split=True, notes="run-42")
    d = cfg.to_dict()
    assert d["verdict_tier"] == "clean" and d["n_cells"] == 10
    assert d["is_clean_tier"] is True and d["notes"] == "run-42"


def test_baseline_and_treatment_accessors():
    cfg = ac.build_cohort(_good_arms(), n_seeds=5)
    assert cfg.baseline_arm().name == "baseline"
    assert [a.name for a in cfg.treatment_arms()] == ["treatment"]
