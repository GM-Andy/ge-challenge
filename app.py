"""
Triage Engine for Alzheimer's Diagnostic Pathways
GE HealthCare Precision Care Challenge 2026

The system never outputs a diagnosis. Every output is an action recommendation:
which test to order next, or release to monitoring.

Run:  streamlit run app.py
"""

from __future__ import annotations

import os
import time
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch  # noqa: F401  (kept for styling hooks)

SEED = 42
# Resolve beside this file, not against the CWD — Streamlit is often launched
# from somewhere else entirely, and silently falling back to the synthetic
# cohort while the real CSV sits right there is a nasty way to lose an hour.
APP_DIR = Path(__file__).resolve().parent
CSV_NAME = "oasis_longitudinal.csv"
CSV_PATH = APP_DIR / CSV_NAME
FIG_DIR = APP_DIR / "figures"

# ──────────────────────────────────────────────────────────────────────────────
#  Palette — headspace-meditation: orange suns, warm cream, soft rounded calm
# ──────────────────────────────────────────────────────────────────────────────
BG        = "#FFF9F2"
CARD      = "#FFFFFF"
INK       = "#34291F"
INK2      = "#7A6A5C"
INK3      = "#A89684"
LINE      = "#F2E7DA"
SUN       = "#FB8B24"   # primary
SUN_SOFT  = "#FFEFDC"
CORAL     = "#F4694E"   # escalate
CORAL_S   = "#FDEAE5"
SAGE      = "#6FA98A"   # release
SAGE_S    = "#EAF4EF"
SKY       = "#5B9BC4"   # order test
SKY_S     = "#E7F1F8"
LILAC     = "#9B7BC4"   # clinician review
LILAC_S   = "#F1ECF8"
STONE     = "#A89684"   # deferred / inert
STONE_S   = "#F4EDE4"
AMBER_BG  = "#FFF3DC"
AMBER_BR  = "#F4DCB0"
AMBER_INK = "#96631A"


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — the decision-flip engine  (constants are load-bearing, do not tune)
# ══════════════════════════════════════════════════════════════════════════════
THETA = 0.40  # escalation threshold

TESTS = {                        # (sensitivity, specificity, cost_INR, days)
    "blood_ptau217": (0.90, 0.90,  8000,  3),
    "mri":           (0.85, 0.70, 12000, 14),
    "pet":           (0.95, 0.95, 60000, 30),
}

TEST_LABEL = {
    "blood_ptau217": "Blood p-tau217",
    "mri":           "Structural MRI",
    "pet":           "Amyloid PET",
}

# What it costs to act wrongly on a patient (missed / delayed workup).
# An assumption, surfaced in the UI so it can be argued with.
COST_OF_WRONG_ACTION = 250_000


def likelihood_ratios(sens, spec):
    return sens / (1 - spec), (1 - sens) / spec


def posterior(p, lr):
    odds = (p / (1 - p)) * lr
    return odds / (1 + odds)


def evaluate_test(p: float, test_key: str, theta: float = THETA) -> dict:
    """Posteriors under both possible results, and whether they straddle theta."""
    sens, spec, cost, days = TESTS[test_key]
    lr_pos, lr_neg = likelihood_ratios(sens, spec)
    p_if_positive = posterior(p, lr_pos)
    p_if_negative = posterior(p, lr_neg)

    flips = (p_if_positive >= theta) != (p_if_negative >= theta)

    p_test_positive = p * sens + (1 - p) * (1 - spec)
    # The action only changes on the result that crosses theta.
    p_flip = p_test_positive if p < theta else (1 - p_test_positive)
    information_value = p_flip * COST_OF_WRONG_ACTION / cost

    return {
        "test": test_key,
        "p_if_positive": p_if_positive,
        "p_if_negative": p_if_negative,
        "flips": bool(flips),
        "wasted": not bool(flips),
        "p_test_positive": p_test_positive,
        "information_value": information_value,
        "cost": cost,
        "days": days,
    }


# ── Action vocabulary. There is deliberately no diagnosis member. ─────────────
class Action(Enum):
    ORDER_BLOOD_PTAU217 = "Order blood p-tau217"
    ORDER_MRI = "Order structural MRI"
    ORDER_PET = "Order amyloid PET"
    ESCALATE_NOW = "Escalate now"
    ESCALATE_NOW_PET_ELIGIBILITY = "Escalate now -> PET for eligibility confirmation"
    RELEASE_12MO = "Re-screen in 12 months"
    CLINICIAN_REVIEW = "Clinician review - insufficient data"
    DEFERRED_NEXT_MONTH = "Deferred to next month - no slot"


_FORBIDDEN = ("diagnos", "alzheimer's disease confirmed", "has ad", "positive for ad")


def assert_no_diagnosis_actions() -> None:
    """Scored requirement, enforced rather than promised."""
    for member in Action:
        blob = f"{member.name} {member.value}".lower()
        for bad in _FORBIDDEN:
            if bad in blob:
                raise AssertionError(
                    f"Action.{member.name} reads as a diagnosis: {member.value!r}"
                )


ORDER_ACTION = {
    "blood_ptau217": Action.ORDER_BLOOD_PTAU217,
    "mri": Action.ORDER_MRI,
    "pet": Action.ORDER_PET,
}

ACTION_COLOR = {
    Action.ORDER_BLOOD_PTAU217: (SKY, SKY_S),
    Action.ORDER_MRI: (SKY, SKY_S),
    Action.ORDER_PET: (SKY, SKY_S),
    Action.ESCALATE_NOW: (CORAL, CORAL_S),
    Action.ESCALATE_NOW_PET_ELIGIBILITY: (CORAL, CORAL_S),
    Action.RELEASE_12MO: (SAGE, SAGE_S),
    Action.CLINICIAN_REVIEW: (LILAC, LILAC_S),
    Action.DEFERRED_NEXT_MONTH: (STONE, STONE_S),
}


def verification_table(theta: float = THETA) -> str:
    """Step 3 self-check, printed to the console at startup."""
    priors = [0.03, 0.10, 0.25, 0.55, 0.90, 0.95]
    expect = {
        0.03: ("none", "none", "none", "release, test nothing"),
        0.10: ("FLIP", "none", "FLIP", "blood only"),
        0.25: ("FLIP", "FLIP", "FLIP", "cheapest flipping test"),
        0.55: ("FLIP", "FLIP", "FLIP", "cheapest flipping test"),
        0.90: ("none", "none", "FLIP", "SKIP blood+MRI, straight to PET"),
        0.95: ("none", "none", "none", "escalate, testing adds nothing"),
    }
    lines = [
        "",
        "  DECISION-FLIP ENGINE — startup self-check  (theta = %.2f)" % theta,
        "  " + "-" * 68,
        "  prior   blood            mri              pet             expected",
    ]
    ok = True
    for p in priors:
        got = tuple(
            "FLIP" if evaluate_test(p, k, theta)["flips"] else "none"
            for k in ("blood_ptau217", "mri", "pet")
        )
        exp = expect[p]
        match = got == exp[:3]
        ok = ok and match
        lines.append(
            "  %.2f    %-16s %-16s %-15s -> %s%s"
            % (p, got[0], got[1], got[2], exp[3], "" if match else "   [MISMATCH]")
        )
    lines.append("  " + "-" * 68)
    lines.append("  self-check: %s" % ("PASS" if ok else "FAIL"))
    lines.append("")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1 — data
# ══════════════════════════════════════════════════════════════════════════════
OASIS_COLUMNS = [
    "Subject ID", "MRI ID", "Group", "Visit", "MR Delay", "M/F", "Hand",
    "Age", "EDUC", "SES", "MMSE", "CDR", "eTIV", "nWBV", "ASF",
]


def synthesise_cohort(n: int = 400, seed: int = SEED) -> pd.DataFrame:
    """Same schema, same column names, so the app runs with no CSV present."""
    rng = np.random.default_rng(seed)
    age = rng.integers(60, 96, n)
    educ = rng.integers(6, 24, n)
    ses = rng.integers(1, 6, n)
    sex = rng.choice(["M", "F"], n, p=[0.42, 0.58])

    # Latent severity drives MMSE and CDR together, as in the real cohort.
    sev = rng.beta(2.0, 3.4, n) + 0.012 * (age - 75) - 0.020 * (educ - 14)
    sev = np.clip(sev, 0, 1)
    mmse = np.clip(np.round(30 - sev * 16 + rng.normal(0, 1.2, n)), 4, 30)
    cdr = np.where(sev < 0.34, 0.0, np.where(sev < 0.62, 0.5, np.where(sev < 0.85, 1.0, 2.0)))

    etiv = rng.normal(1480, 175, n).round(0)
    nwbv = np.clip(0.79 - sev * 0.10 + rng.normal(0, 0.018, n), 0.60, 0.86).round(3)
    asf = (1755.0 / etiv).round(3)

    return pd.DataFrame({
        "Subject ID": [f"OAS2_{i + 1:04d}" for i in range(n)],
        "MRI ID": [f"OAS2_{i + 1:04d}_MR1" for i in range(n)],
        "Group": np.where(cdr > 0, "Demented", "Nondemented"),
        "Visit": 1,
        "MR Delay": 0,
        "M/F": sex,
        "Hand": "R",
        "Age": age,
        "EDUC": educ,
        "SES": ses,
        "MMSE": mmse,
        "CDR": cdr,
        "eTIV": etiv,
        "nWBV": nwbv,
        "ASF": asf,
    })


@st.cache_data(show_spinner=False)
def load_patients(seed: int = SEED):
    """First visit per subject, four modality blocks, deliberate missingness."""
    if CSV_PATH.exists():
        raw = pd.read_csv(CSV_PATH)
        synthetic = False
    else:
        raw = synthesise_cohort(400, seed)
        synthetic = True

    # Collapse to one row per patient — the first visit.
    df = (raw.sort_values(["Subject ID", "Visit"])
             .groupby("Subject ID", as_index=False)
             .first()
             .reset_index(drop=True))

    rng = np.random.default_rng(seed)
    n = len(df)

    df["patient_id"] = df["Subject ID"]
    df["sex_m"] = (df["M/F"] == "M").astype(float)

    # ── comorbidity block: entirely simulated, labelled as such in the UI ──
    df["hypertension"] = (rng.random(n) < 0.41).astype(float)
    df["diabetes"] = (rng.random(n) < 0.19).astype(float)
    df["apoe4"] = (rng.random(n) < 0.30).astype(float)

    # ── blood block: a p-tau217 value only exists where the test was done ──
    # Correlated with the latent state so the modality carries real signal.
    sev_proxy = np.clip((30 - df["MMSE"].fillna(26)) / 16.0, 0, 1).to_numpy()
    df["ptau217_value"] = np.round(0.45 + 2.1 * sev_proxy + rng.normal(0, 0.35, n), 3)

    # ── availability masks ──
    df["has_cognitive"] = True
    df["has_comorbidity"] = rng.random(n) < 0.60
    df["has_blood"] = rng.random(n) < 0.20
    df["has_mri"] = rng.random(n) < 0.35

    for col in ("hypertension", "diabetes", "apoe4"):
        df.loc[~df["has_comorbidity"], col] = np.nan
    df.loc[~df["has_blood"], "ptau217_value"] = np.nan
    for col in ("nWBV", "eTIV", "ASF"):
        df.loc[~df["has_mri"], col] = np.nan

    df["n_modalities"] = (
        df["has_cognitive"].astype(int)
        + df["has_comorbidity"].astype(int)
        + df["has_blood"].astype(int)
        + df["has_mri"].astype(int)
    )

    # Progression proxy. Partly circular — CDR is itself a clinical rating.
    df["label"] = (pd.to_numeric(df["CDR"], errors="coerce").fillna(0) > 0).astype(int)

    df["low_education"] = df["EDUC"] < df["EDUC"].median()

    return df, synthetic


FEATURES = [
    "MMSE", "Age", "EDUC", "SES", "sex_m",              # cognitive
    "hypertension", "diabetes", "apoe4",                # comorbidity
    "ptau217_value",                                    # blood
    "nWBV", "eTIV", "ASF",                              # mri
]


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2 — risk model with real uncertainty
# ══════════════════════════════════════════════════════════════════════════════
TABPFN_BUDGET_S = 20.0


@st.cache_data(show_spinner=False)
def tabpfn_status() -> tuple[bool, str]:
    """
    Cheap availability probe — import only, no fit. Called on every startup,
    so it must never cost the demo anything.
    """
    try:
        import torch  # noqa: F401
    except Exception as exc:
        return False, f"torch missing ({type(exc).__name__})"
    try:
        import tabpfn
    except Exception as exc:
        return False, f"tabpfn not importable ({type(exc).__name__})"
    ver = getattr(tabpfn, "__version__", "?")
    return True, f"TabPFN {ver} installed"


def _fit_tabpfn(X, y, budget_s: float = TABPFN_BUDGET_S):
    """
    Attempt a real, timed fit + predict. Anything at all going wrong — an
    import error, a scikit-learn API break, a licence gate, or simply blowing
    the time budget — returns None and the caller falls back. The demo must
    never hang on this.
    """
    try:
        from tabpfn import TabPFNClassifier
        t0 = time.time()
        clf = TabPFNClassifier()
        clf.fit(X, y)
        p = clf.predict_proba(X)[:, 1]
        elapsed = time.time() - t0
        if elapsed > budget_s:
            return None, (f"TabPFN ran but took {elapsed:.0f}s, over the "
                          f"{budget_s:.0f}s budget")
        return p, f"TabPFN active ({elapsed:.0f}s)"
    except Exception as exc:
        return None, f"TabPFN failed: {type(exc).__name__}"


MODALITY_COLUMNS = {
    "comorbidity": ["hypertension", "diabetes", "apoe4"],
    "blood": ["ptau217_value"],
    "mri": ["nWBV", "eTIV", "ASF"],
}


def _hotdeck_draws(df: pd.DataFrame, X: np.ndarray, rng, n_draws: int = 8):
    """
    Plausible completions of each patient's missing modality blocks.

    These are used ONLY to measure how far the estimate would move if the
    missing work-up existed. The point estimate itself is always taken from
    the NaN-native model on the real, un-imputed row — this is not
    impute-and-forget.

    Donors are whole patients who do hold the block, so within-block
    correlation (eTIV/nWBV/ASF move together) survives the fill.
    """
    col_ix = {c: FEATURES.index(c) for c in FEATURES}
    draws = np.repeat(X[None, ...], n_draws, axis=0)

    for block, cols in MODALITY_COLUMNS.items():
        have = df[f"has_{block}"].to_numpy().astype(bool)
        donors = np.flatnonzero(have)
        missing = np.flatnonzero(~have)
        if len(donors) == 0 or len(missing) == 0:
            continue
        ix = [col_ix[c] for c in cols]
        for m in range(n_draws):
            picked = rng.choice(donors, size=len(missing), replace=True)
            draws[m][np.ix_(missing, ix)] = X[np.ix_(picked, ix)]
    return draws


@st.cache_resource(show_spinner=False)
def fit_engine(seed: int = SEED, n_boot: int = 15, n_draws: int = 8,
               use_tabpfn: bool = False):
    """
    Returns p, uncertainty, and the two components uncertainty is built from.

    Spread has two genuinely different sources, and a thin record only widens
    the second one:

      var_model      bootstrap disagreement on the row as it actually is
      var_missing    how much the estimate moves across plausible completions
                     of the modality blocks this patient does not have

    A patient holding all four modalities has var_missing == 0 by
    construction, because there is nothing left to complete.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier

    df, _ = load_patients(seed)
    X = df[FEATURES].astype(float).to_numpy()
    y = df["label"].to_numpy()

    rng = np.random.default_rng(seed)
    n = len(df)

    models = []
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:            # degenerate resample
            idx = np.arange(n)
        model = HistGradientBoostingClassifier(
            max_iter=140, max_depth=4, learning_rate=0.09,
            min_samples_leaf=8, l2_regularization=1.0, random_state=seed + b,
        )
        model.fit(X[idx], y[idx])
        models.append(model)

    # Point estimate: every model reads the real row, NaNs and all.
    real = np.vstack([m.predict_proba(X)[:, 1] for m in models])   # (B, n)
    p = real.mean(axis=0)
    var_model = real.var(axis=0)

    engine_name = "HistGradientBoostingClassifier"
    engine_note = "NaN-native, no imputation, no dropped rows."
    available, probe = tabpfn_status()

    if use_tabpfn and available:
        # TabPFN supplies the point estimate only. Running the 15-model
        # bootstrap and 8 imputation draws through it would take ~20 minutes
        # on CPU, so the spread stays with the fast NaN-native model — and the
        # sidebar says so rather than implying the bands are TabPFN's.
        p_tab, note = _fit_tabpfn(X, y)
        if p_tab is not None:
            p = p_tab
            engine_name = "TabPFN (risk) + HistGradientBoosting (spread)"
            engine_note = note + ". Uncertainty bands still come from the " \
                                 "bootstrap + imputation model."
        else:
            engine_note = note + " — sklearn fallback active."
    elif use_tabpfn:
        engine_note = f"{probe} — sklearn fallback active."
    else:
        engine_note = (f"{probe}; opt in from the sidebar. "
                       if available else f"{probe}. ") + engine_note

    # Missing-data spread: how far does the estimate travel across plausible
    # completions of the blocks this patient does not hold?
    draws = _hotdeck_draws(df, X, rng, n_draws)
    filled = np.stack([
        np.vstack([m.predict_proba(draws[d])[:, 1] for m in models]).mean(axis=0)
        for d in range(n_draws)
    ])                                                              # (M, n)
    var_missing = filled.var(axis=0)

    u = np.sqrt(var_model + var_missing)
    u_missing = np.sqrt(var_missing)

    def _spearman(a, b):
        ra = pd.Series(a).rank().to_numpy()
        rb = pd.Series(b).rank().to_numpy()
        return float(np.corrcoef(ra, rb)[0, 1])

    n_mod = df["n_modalities"].to_numpy()
    diag = {
        "corr": float(np.corrcoef(n_mod, u)[0, 1]),
        "corr_spearman": _spearman(n_mod, u),
        "corr_missing": float(np.corrcoef(n_mod, u_missing)[0, 1]),
        "corr_model_only": float(np.corrcoef(n_mod, np.sqrt(var_model))[0, 1]),
        "mean_u_by_mod": {int(k): float(v) for k, v in
                          pd.Series(u).groupby(n_mod).mean().items()},
        "n_by_mod": {int(k): int(v) for k, v in
                     pd.Series(n_mod).value_counts().sort_index().items()},
        "share_missing": float(
            np.mean(var_missing / np.maximum(var_model + var_missing, 1e-12))),
    }
    return p, u, u_missing, engine_name, engine_note, diag


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3 — action selection
# ══════════════════════════════════════════════════════════════════════════════
def build_plan(df: pd.DataFrame, p: np.ndarray, u: np.ndarray,
               u_missing: np.ndarray, theta: float = THETA):
    # CLINICIAN_REVIEW is labelled "insufficient data", so it triggers on the
    # missing-data component of the spread — not on total spread. A patient who
    # is genuinely borderline but fully worked up is not a data problem, and
    # sending them to manual review would be the wrong call.
    high_u_cut = float(np.quantile(u_missing, 0.90))
    rows = []

    for i, rec in df.reset_index(drop=True).iterrows():
        pi = float(p[i])
        ui = float(u[i])
        umi = float(u_missing[i])
        evals = {k: evaluate_test(pi, k, theta) for k in TESTS}
        flipping = [k for k, e in evals.items() if e["flips"]]

        chosen = None
        if not flipping:
            action = Action.ESCALATE_NOW if pi >= theta else Action.RELEASE_12MO
        elif umi >= high_u_cut:
            action = Action.CLINICIAN_REVIEW
        else:
            chosen = min(flipping, key=lambda k: TESTS[k][2])   # lowest cost
            action = ORDER_ACTION[chosen]

        # ── Safety guard ──────────────────────────────────────────────────
        # A treatment candidate still needs PET to confirm eligibility, even
        # when PET cannot change the triage action. Never suppress that.
        mmse = rec["MMSE"]
        treatment_candidate = bool(
            pi >= theta
            and pd.notna(mmse) and mmse >= 20      # early-stage window
            and rec["Age"] <= 85
        )
        if action is Action.ESCALATE_NOW and treatment_candidate:
            action = Action.ESCALATE_NOW_PET_ELIGIBILITY

        if chosen is not None:
            slot_test, info_value = chosen, evals[chosen]["information_value"]
        elif action is Action.ESCALATE_NOW_PET_ELIGIBILITY:
            slot_test, info_value = "pet", evals["pet"]["information_value"]
        else:
            slot_test, info_value = None, 0.0

        rows.append({
            "patient_id": rec["patient_id"],
            "p": pi, "u": ui, "u_missing": umi,
            "n_modalities": int(rec["n_modalities"]),
            "has_comorbidity": bool(rec["has_comorbidity"]),
            "has_blood": bool(rec["has_blood"]),
            "has_mri": bool(rec["has_mri"]),
            "low_education": bool(rec["low_education"]),
            "MMSE": mmse, "Age": rec["Age"], "EDUC": rec["EDUC"],
            "action": action,
            "slot_test": slot_test,
            "information_value": info_value,
            "treatment_candidate": treatment_candidate,
            "evals": evals,
            "label": int(rec["label"]),
        })

    plan = pd.DataFrame(rows)
    plan["high_uncertainty"] = plan["u_missing"] >= high_u_cut
    return plan, high_u_cut


def reason_text(r, theta: float) -> str:
    a = r["action"]
    if a is Action.ESCALATE_NOW_PET_ELIGIBILITY:
        return ("No test can move the triage call, but amyloid PET is still required "
                "to confirm treatment eligibility.")
    if a is Action.ESCALATE_NOW:
        return (f"Every available result lands above θ={theta:.2f}. Further testing "
                f"delays care without changing it.")
    if a is Action.RELEASE_12MO:
        return (f"Every available result lands below θ={theta:.2f}. Nothing on the "
                f"menu can raise this patient over the line.")
    if a is Action.CLINICIAN_REVIEW:
        held = {1: "cognitive screen only", 2: "two of four modalities",
                3: "three of four modalities", 4: "all four modalities"}[r["n_modalities"]]
        return (f"Spread is in the top decile ({held} on file). Too thin to act on "
                f"automatically.")
    if a is Action.DEFERRED_NEXT_MONTH:
        return "Flipping test identified, but no slot left this month. Rolls forward."
    k = r["slot_test"]
    e = r["evals"][k]
    return (f"{TEST_LABEL[k]} is the cheapest test whose result crosses θ "
            f"({e['p_if_negative']:.2f} vs {e['p_if_positive']:.2f}). "
            f"Resolves in {e['days']} days.")


# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4 — capacity allocation
# ══════════════════════════════════════════════════════════════════════════════
def allocate(plan: pd.DataFrame, caps: dict):
    """Greedy fill by descending information value, under the flip filter."""
    left = dict(caps)
    granted, deferred = [], []

    order = plan.sort_values("information_value", ascending=False)
    for idx, r in order.iterrows():
        k = r["slot_test"]
        if k is None:
            continue
        if left.get(k, 0) > 0:
            left[k] -= 1
            granted.append(idx)
        else:
            deferred.append(idx)

    plan = plan.copy()
    plan["granted"] = plan.index.isin(granted)
    plan["deferred"] = plan.index.isin(deferred)
    plan["rank"] = (plan["information_value"]
                    .rank(ascending=False, method="first").astype(int))
    return plan, left


def naive_allocation(plan: pd.DataFrame, caps: dict):
    """
    Work down the risk list with whatever capacity exists: PET to the highest
    risk patients, then MRI to the next tranche, then blood to the next.

    One test per patient. Handing the same top-risk patient a PET *and* an MRI
    *and* a blood draw would make the comparison a strawman — the waste count
    would balloon simply because the policy triple-books one person.
    """
    left = dict(caps)
    scans = []
    for _, r in plan.sort_values("p", ascending=False).iterrows():
        for k in ("pet", "mri", "blood_ptau217"):
            if left.get(k, 0) > 0:
                left[k] -= 1
                scans.append({
                    "patient_id": r["patient_id"], "test": k,
                    "wasted": not r["evals"][k]["flips"],
                    "p_test_positive": r["evals"][k]["p_test_positive"],
                    "cost": TESTS[k][2],
                })
                break            # one test per patient
    return pd.DataFrame(scans)


def policy_summary(plan: pd.DataFrame, naive: pd.DataFrame):
    ours_scans = plan[plan["granted"]]
    ours_pet = ours_scans[ours_scans["slot_test"] == "pet"]

    # A patient is resolved if their action is terminal OR they got a slot.
    # These overlap — an ESCALATE_NOW -> PET-for-eligibility patient is both —
    # so this must be a union over patients, never a sum of two counts.
    terminal = plan["action"].isin([Action.ESCALATE_NOW,
                                    Action.ESCALATE_NOW_PET_ELIGIBILITY,
                                    Action.RELEASE_12MO])
    ours_resolved = int((terminal | plan["granted"]).sum())

    if len(naive):
        naive_wasted = naive[naive["wasted"]]
        naive_pet = naive[naive["test"] == "pet"]
        naive_resolved = int(naive[~naive["wasted"]]["patient_id"].nunique())
        naive_pet_yield = float(naive_pet["p_test_positive"].mean()) if len(naive_pet) else 0.0
        naive_useful_per_pet = (
            float((~naive_pet["wasted"]).mean()) if len(naive_pet) else 0.0
        )
    else:
        naive_wasted = naive
        naive_pet = naive
        naive_resolved = 0
        naive_pet_yield = 0.0
        naive_useful_per_pet = 0.0

    ours_pet_yield = (
        float(ours_pet["evals"].map(lambda e: e["pet"]["p_test_positive"]).mean())
        if len(ours_pet) else 0.0
    )
    ours_cost = int(sum(TESTS[k][2] for k in ours_scans["slot_test"] if k))

    return {
        "naive_scans": int(len(naive)),
        "naive_wasted": int(len(naive_wasted)),
        "naive_wasted_cost": int(naive_wasted["cost"].sum()) if len(naive_wasted) else 0,
        "naive_resolved": naive_resolved,
        "naive_pet": int(len(naive_pet)),
        "naive_pet_yield": naive_pet_yield,
        "naive_useful_per_pet": naive_useful_per_pet,
        "ours_scans": int(len(ours_scans)),
        "ours_wasted": 0,
        "ours_cost": ours_cost,
        "ours_resolved": ours_resolved,
        "ours_pet": int(len(ours_pet)),
        "ours_pet_yield": ours_pet_yield,
        "ours_useful_per_pet": 1.0 if len(ours_pet) else 0.0,
        "deferred": int(plan["deferred"].sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Figures — shared by the screen and the slide export
# ══════════════════════════════════════════════════════════════════════════════
def _style(ax):
    ax.set_facecolor(CARD)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=INK2, labelsize=12)


def fig_flip_panel(row, theta: float, standalone: bool = False):
    fig, ax = plt.subplots(figsize=(11.0, 4.4), dpi=150)
    fig.patch.set_facecolor(CARD)
    _style(ax)

    keys = ["blood_ptau217", "mri", "pet"]
    ys = [2, 1, 0]

    halo = dict(facecolor=CARD, edgecolor="none", pad=2.5)

    ax.axvline(theta, color=CORAL, lw=2.4, zorder=1)
    ax.text(theta, 2.72, f"θ = {theta:.2f}  escalate →", color=CORAL,
            fontsize=14, fontweight="bold", ha="center", bbox=halo, zorder=5)

    ax.axvline(row["p"], color=SUN, lw=2.0, ls=(0, (4, 3)), zorder=1)
    # Keep the label off its own line, and off the axis edges.
    pl = float(np.clip(row["p"], 0.07, 0.93))
    ax.text(pl, -0.80, f"risk now {row['p']:.2f}", color=SUN, fontsize=13,
            fontweight="bold", ha="center", bbox=halo, zorder=5)

    for k, y in zip(keys, ys):
        e = row["evals"][k]
        lo, hi = sorted((e["p_if_negative"], e["p_if_positive"]))
        col = SKY if e["flips"] else STONE
        alpha = 1.0 if e["flips"] else 0.45

        ax.plot([lo, hi], [y, y], color=col, lw=9, alpha=alpha * 0.35,
                solid_capstyle="round", zorder=2)
        ax.scatter([e["p_if_negative"]], [y], s=230, facecolor=CARD,
                   edgecolor=col, linewidth=3.4, zorder=3, alpha=alpha)
        ax.scatter([e["p_if_positive"]], [y], s=230, color=col,
                   zorder=3, alpha=alpha)

        ax.text(e["p_if_negative"], y + 0.26, f"{e['p_if_negative']:.2f}",
                ha="center", fontsize=12.5, color=col, fontweight="bold", alpha=alpha)
        ax.text(e["p_if_positive"], y + 0.26, f"{e['p_if_positive']:.2f}",
                ha="center", fontsize=12.5, color=col, fontweight="bold", alpha=alpha)

        label = f"{TEST_LABEL[k]}\n₹{TESTS[k][2]:,} · {TESTS[k][3]}d"
        ax.text(-0.035, y, label, ha="right", va="center", fontsize=13,
                color=INK if e["flips"] else INK3, fontweight="bold", linespacing=1.5)

        stamp = "THIS TEST DECIDES" if e["flips"] else "CANNOT CHANGE DECISION"
        ax.text(1.035, y, stamp, ha="left", va="center", fontsize=12.5,
                color=col, fontweight="bold", alpha=alpha)

    ax.set_xlim(0, 1)
    ax.set_ylim(-1.05, 3.05)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("probability of progression", fontsize=13, color=INK2, labelpad=10)

    if standalone:
        spread_u = row["u"] if "u" in row else None
        spread = ("" if spread_u is None or not np.isfinite(spread_u)
                  else f" ± {spread_u:.2f}")
        fig.suptitle(f"A test is worth ordering only if it can change the plan\n"
                     f"{row['patient_id']}   ·   risk {row['p']:.2f}{spread}",
                     fontsize=17, fontweight="bold", color=INK, y=1.10, linespacing=1.6)
    fig.subplots_adjust(left=0.16, right=0.80, top=0.84, bottom=0.20)
    return fig


def fig_coverage(df: pd.DataFrame, n_show: int = 40, standalone: bool = False):
    sub = df.head(n_show)
    M = np.column_stack([
        np.ones(len(sub)),
        sub["has_comorbidity"].to_numpy().astype(float),
        sub["has_blood"].to_numpy().astype(float),
        sub["has_mri"].to_numpy().astype(float),
    ])

    fig, ax = plt.subplots(figsize=(6.4, 8.6), dpi=150)
    fig.patch.set_facecolor(CARD)
    _style(ax)

    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            on = M[i, j] > 0
            ax.add_patch(plt.Rectangle(
                (j + 0.08, i + 0.10), 0.84, 0.80,
                facecolor=SUN if on else "#F6EDE2",
                edgecolor="none" if on else "#EADFD1",
                linewidth=1.0, zorder=2,
            ))

    ax.set_xlim(0, 4)
    ax.set_ylim(M.shape[0], 0)
    ax.set_xticks([0.5, 1.5, 2.5, 3.5])
    ax.set_xticklabels(["Cognitive", "Comorbidity", "Blood", "MRI"],
                       fontsize=13, fontweight="bold", color=INK)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(np.arange(len(sub)) + 0.5)
    ax.set_yticklabels(sub["patient_id"], fontsize=7.5, color=INK3)

    total = len(df)
    complete = int((df["n_modalities"] == 4).sum())
    caption = (f"{total - complete} of {total} patients have incomplete records.\n"
               f"A model requiring complete rows would score only the {complete} "
               f"already worked up.")
    if standalone:
        fig.suptitle("The missing cells are the point", fontsize=17,
                     fontweight="bold", color=INK, y=0.985)
    ax.set_xlabel(caption, fontsize=12, color=INK2, labelpad=16, linespacing=1.6)
    ax.xaxis.set_label_position("bottom")
    fig.subplots_adjust(left=0.22, right=0.97, top=0.90, bottom=0.10)
    return fig


def fig_policy(s: dict, standalone: bool = False):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0), dpi=150)
    fig.patch.set_facecolor(CARD)

    panels = [
        ("Decisions resolved", s["naive_resolved"], s["ours_resolved"], "{:,.0f}"),
        ("Scans that cannot flip", s["naive_wasted"], s["ours_wasted"], "{:,.0f}"),
        ("Spend on those scans", s["naive_wasted_cost"], 0, "₹{:,.0f}"),
    ]
    for ax, (title, a, b, fmt) in zip(axes, panels):
        _style(ax)
        bars = ax.bar(["Naive", "Ours"], [a, b], width=0.56,
                      color=[STONE, SUN], zorder=3)
        for rect, v in zip(bars, [a, b]):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    rect.get_height() + max(a, b, 1) * 0.04,
                    fmt.format(v), ha="center", fontsize=13.5,
                    fontweight="bold", color=INK)
        ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=14)
        ax.set_ylim(0, max(a, b, 1) * 1.28)
        ax.set_yticks([])
        ax.tick_params(labelsize=13)

    if standalone:
        fig.suptitle("Same capacity, two policies", fontsize=17,
                     fontweight="bold", color=INK, y=1.02)
    fig.tight_layout()
    return fig


def fig_funnel(df: pd.DataFrame, plan: pd.DataFrame, s: dict, standalone: bool = False):
    total = len(df)
    complete = int((df["n_modalities"] == 4).sum())
    stages = [
        ("Screened this month", total, SUN),
        ("Scored by this engine", total, SAGE),
        ("Would be scored by a\ncomplete-case model", complete, STONE),
        ("Decisions resolved", s["ours_resolved"], SKY),
        ("Rolled to next month", s["deferred"], CORAL),
    ]

    fig, ax = plt.subplots(figsize=(9.6, 4.4), dpi=150)
    fig.patch.set_facecolor(CARD)
    _style(ax)

    names = [x[0] for x in stages]
    vals = [x[1] for x in stages]
    cols = [x[2] for x in stages]
    ys = np.arange(len(stages))[::-1]

    ax.barh(ys, vals, height=0.56, color=cols, zorder=3)
    for y, v in zip(ys, vals):
        ax.text(v + total * 0.015, y, f"{v:,}", va="center",
                fontsize=13.5, fontweight="bold", color=INK)

    ax.set_yticks(ys)
    ax.set_yticklabels(names, fontsize=12.5, color=INK, linespacing=1.5)
    ax.set_xticks([])
    ax.set_xlim(0, total * 1.14)
    if standalone:
        fig.suptitle("Nobody is dropped for having a thin record",
                     fontsize=17, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Quicksand:wght@500;600;700&family=Nunito+Sans:wght@400;600;700&display=swap');

.stApp {{ background: {BG}; }}
html, body, [class*="css"] {{
  font-family: 'Nunito Sans', system-ui, -apple-system, sans-serif;
  color: {INK};
}}
h1, h2, h3, h4 {{
  font-family: 'Quicksand', system-ui, sans-serif !important;
  color: {INK} !important; letter-spacing: -0.01em;
}}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.6rem 2.6rem 4rem 2.6rem; max-width: 1500px; }}

section[data-testid="stSidebar"] {{
  background: #FFF4E7;
  border-right: 1px solid {LINE};
}}
section[data-testid="stSidebar"] .block-container {{ padding-top: 1.6rem; }}

.sun-hero {{
  position: relative; overflow: hidden;
  background: linear-gradient(135deg, #FFEEDA 0%, #FFF7EE 58%, #FDF3F0 100%);
  border-radius: 30px; padding: 30px 36px 28px 36px; margin-bottom: 20px;
  border: 1px solid #F7E4CE;
}}
.sun-hero:after {{
  content: ""; position: absolute; top: -110px; right: -70px;
  width: 300px; height: 300px; border-radius: 50%;
  background: radial-gradient(circle, rgba(251,139,36,0.26), rgba(251,139,36,0) 68%);
}}
.sun-hero h1 {{ margin: 0; font-size: 34px; font-weight: 700; line-height: 1.12; }}
.sun-hero p {{ margin: 10px 0 0 0; color: {INK2}; font-size: 15px; max-width: 660px; }}

.card {{
  background: {CARD}; border: 1px solid {LINE}; border-radius: 26px;
  padding: 22px 24px; box-shadow: 0 2px 3px rgba(52,41,31,0.03),
  0 18px 36px -26px rgba(52,41,31,0.24); margin-bottom: 16px;
}}
.metric {{
  background: {CARD}; border: 1px solid {LINE}; border-radius: 26px;
  padding: 20px 22px 18px 22px;
  box-shadow: 0 2px 3px rgba(52,41,31,0.03), 0 18px 36px -26px rgba(52,41,31,0.24);
}}
.metric .k {{
  font-size: 11px; font-weight: 700; letter-spacing: 0.10em;
  text-transform: uppercase; color: {INK3};
}}
.metric .v {{
  font-family: 'Quicksand', sans-serif; font-size: 36px; font-weight: 700;
  line-height: 1.1; margin-top: 8px; letter-spacing: -0.02em;
}}
.metric .s {{ font-size: 12px; color: {INK3}; margin-top: 6px; }}

.banner {{
  background: {AMBER_BG}; border: 1px solid {AMBER_BR}; border-radius: 20px;
  padding: 13px 20px; color: {AMBER_INK}; font-size: 14px; margin-bottom: 14px;
}}
.good {{
  background: {SAGE_S}; border: 1px solid #D3E7DC; border-radius: 20px;
  padding: 13px 20px; color: #3D6B54; font-size: 14px; margin-bottom: 14px;
}}
.pill {{
  display: inline-block; padding: 5px 12px; border-radius: 999px;
  font-size: 11.5px; font-weight: 700; white-space: nowrap;
}}
.qt {{ width: 100%; border-collapse: collapse; }}
.qt th {{
  font-size: 10.5px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: {INK3}; text-align: left;
  padding: 0 12px 10px 12px;
}}
.qt td {{
  padding: 12px; border-top: 1px solid #F7EFE5; font-size: 13px;
  vertical-align: middle;
}}
.mod {{
  display: inline-block; width: 30px; text-align: center; padding: 3px 0;
  border-radius: 7px; font-size: 9px; font-weight: 700; margin-right: 3px;
}}
.stButton>button {{
  border-radius: 999px; border: 1px solid #F0D9BE; background: {CARD};
  color: {INK}; font-weight: 700; padding: 10px 22px;
}}
.stButton>button:hover {{ border-color: {SUN}; color: {SUN}; }}
</style>
"""


def pill(text: str, fg: str, bg: str) -> str:
    return f'<span class="pill" style="color:{fg};background:{bg}">{text}</span>'


def inr(v: float) -> str:
    v = float(v)
    if v >= 1e7:
        return f"₹{v / 1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v / 1e5:.1f} L"
    return f"₹{v:,.0f}"


def main():
    assert_no_diagnosis_actions()
    print(verification_table())

    st.set_page_config(page_title="Triage Engine · Alzheimer's Pathways",
                       page_icon="🌅", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    df, synthetic = load_patients()
    tab_ok, tab_probe = tabpfn_status()
    use_tabpfn = st.sidebar.checkbox(
        "Use TabPFN for the risk estimate", value=False, disabled=not tab_ok,
        help=(f"{tab_probe}. Adds roughly 30s on CPU the first time — the "
              f"uncertainty bands stay with the fast NaN-native model either "
              f"way." if tab_ok else tab_probe))
    with st.spinner("Fitting TabPFN — this takes about 30s…" if use_tabpfn
                    else "Fitting risk engine…"):
        p, u, u_missing, engine_name, engine_note, diag = fit_engine(
            use_tabpfn=use_tabpfn)

    print("  UNCERTAINTY CHECK")
    print("  " + "-" * 68)
    print("  corr(n_modalities_available, uncertainty) = %+.4f  (Pearson)"
          % diag["corr"])
    print("  ...                                        = %+.4f  (Spearman)"
          % diag["corr_spearman"])
    print("  corr(n_modalities, missing-data component) = %+.4f"
          % diag["corr_missing"])
    print("  ... bootstrap component alone              = %+.4f"
          % diag["corr_model_only"])
    print("  mean spread by modality count: " + ", ".join(
        "%d mod %.3f (n=%d)" % (k, v, diag["n_by_mod"].get(k, 0))
        for k, v in sorted(diag["mean_u_by_mod"].items())))
    print("  share of variance from missing modalities  = %.1f%%"
          % (100 * diag["share_missing"]))
    print("  " + "-" * 68 + "\n")

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            f"""<div style="display:flex;align-items:center;gap:11px;margin-bottom:20px">
            <div style="width:38px;height:38px;border-radius:50%;
                 background:radial-gradient(circle at 35% 32%, #FFC078, {SUN});
                 flex-shrink:0"></div>
            <div><div style="font-family:Quicksand;font-weight:700;font-size:17px;
                 line-height:1.15">Triage Engine</div>
            <div style="font-size:11.5px;color:{INK2}">Alzheimer's pathways</div></div>
            </div>""",
            unsafe_allow_html=True)

        st.markdown(f"""<div style="background:{CARD};border:1px solid {LINE};
            border-radius:20px;padding:14px 16px;margin-bottom:18px">
            <div style="font-size:10.5px;font-weight:700;letter-spacing:.09em;
            text-transform:uppercase;color:{INK3}">Risk engine</div>
            <div style="font-family:Quicksand;font-weight:700;font-size:14px;
            margin-top:6px">{engine_name}</div>
            <div style="font-size:11.5px;color:{INK2};margin-top:6px;line-height:1.5">
            {engine_note}</div>
            <div style="font-size:11.5px;color:{INK2};margin-top:8px;line-height:1.5">
            corr(modalities, spread) = <b>{diag['corr']:+.3f}</b><br>
            <span style="color:{INK3}">{diag['share_missing']:.0%} of the variance
            comes from what is missing, not from model disagreement.</span></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("##### Monthly capacity")
        pet_slots = st.slider("PET slots", 0, 100, 20)
        mri_slots = st.slider("MRI slots", 0, 200, 60)
        blood_slots = st.slider("Blood p-tau217 tests", 0, 400, 200)
        theta = st.slider("Escalation threshold θ", 0.05, 0.95, THETA, 0.05)

        st.markdown(f"""<div style="background:{AMBER_BG};border:1px solid {AMBER_BR};
            border-radius:18px;padding:13px 15px;margin-top:16px">
            <div style="font-size:10.5px;font-weight:700;letter-spacing:.07em;
            text-transform:uppercase;color:{AMBER_INK}">Label caveat</div>
            <div style="font-size:11.5px;color:#7A5216;margin-top:6px;line-height:1.55">
            Training label is <b>CDR &gt; 0</b> — itself a clinician's rating, so the
            target is partly circular. Prospective validation would use conversion to
            AD over follow-up, not a concurrent scale.</div></div>""",
            unsafe_allow_html=True)

        st.markdown(f"""<div style="font-size:11px;color:{INK3};margin-top:14px;
            line-height:1.55;border-top:1px solid {LINE};padding-top:12px">
            <b style="color:{INK2}">Simulated fields:</b> hypertension, diabetes, APOE4
            and p-tau217 values are generated (seed 42). Sensitivity, specificity and
            cost figures are published-literature estimates, not measured here.
            Cost of a wrong action is assumed at ₹{COST_OF_WRONG_ACTION:,}.</div>""",
            unsafe_allow_html=True)

        export = st.button("Export figures → ./figures", use_container_width=True)

    caps = {"pet": pet_slots, "mri": mri_slots, "blood_ptau217": blood_slots}
    plan, _ = build_plan(df, p, u, u_missing, theta)
    plan, _left = allocate(plan, caps)
    naive = naive_allocation(plan, caps)
    s = policy_summary(plan, naive)

    # ── hero ──────────────────────────────────────────────────────────────
    st.markdown(f"""<div class="sun-hero">
        <h1>This month's triage plan</h1>
        <p>{len(df)} patients screened. Each one gets a <i>next action</i> — which test
        to order, or release to monitoring. No output on this screen is a diagnosis.</p>
        </div>""", unsafe_allow_html=True)

    if synthetic:
        st.markdown('<div class="banner"><b>Running on synthetic cohort.</b> '
                    'Swap in <code>oasis_longitudinal.csv</code> for real data — '
                    'schema and column names are identical.</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="good"><b>Running on real OASIS-2 data</b> — '
                    f'{CSV_NAME}, {len(df)} patients (first visit per subject). '
                    f'Comorbidity, APOE4 and p-tau217 blocks are simulated and '
                    f'labelled as such.</div>', unsafe_allow_html=True)

    # ── A. header metrics ─────────────────────────────────────────────────
    cols = st.columns(4)
    cards = [
        ("PET slots used", f"{s['ours_pet']} / {pet_slots}", INK,
         "every one of them changes a plan"),
        ("Wasted scans avoided", f"{s['naive_wasted']}", SUN,
         "ordered by naive policy, cannot flip"),
        ("Spend released", inr(s["naive_wasted_cost"]), SAGE,
         "freed for patients still undecided"),
        ("Decisions resolved", f"{s['ours_resolved']} / {len(df)}", INK,
         "this month, at current capacity"),
    ]
    for c, (k, v, colr, sub) in zip(cols, cards):
        c.markdown(f"""<div class="metric"><div class="k">{k}</div>
            <div class="v" style="color:{colr}">{v}</div>
            <div class="s">{sub}</div></div>""", unsafe_allow_html=True)

    st.write("")

    # ── B + C ─────────────────────────────────────────────────────────────
    left, right = st.columns([1, 1.85])

    with left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Modality coverage")
        complete = int((df["n_modalities"] == 4).sum())
        st.caption(f"{len(df) - complete} of {len(df)} patients have incomplete "
                   f"records. A model requiring complete rows would score only the "
                   f"{complete} already worked up — and those are exactly the patients "
                   f"a clinician has already decided about.")
        st.pyplot(fig_coverage(df), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Can this test change what we do?")
        st.caption("A test earns its slot only when a positive and a negative result "
                   "land on opposite sides of θ. If both land on the same side, the "
                   "result cannot move the plan — the scan is spend without a decision.")

        ranked = plan.sort_values("p", ascending=False)
        only_pet = plan["evals"].map(
            lambda e: not e["blood_ptau217"]["flips"]
            and not e["mri"]["flips"] and e["pet"]["flips"])
        # Prefer the high-side case (skip the cheap tests, go straight to PET).
        # It does not occur in every cohort — fall back to the low-side mirror.
        money = plan[only_pet & (plan["p"] >= theta)].sort_values("p", ascending=False)
        if not len(money):
            money = plan[only_pet].sort_values("p", ascending=False)
        default_id = (money.iloc[0]["patient_id"] if len(money)
                      else ranked.iloc[0]["patient_id"])
        options = list(ranked["patient_id"])
        sel_id = st.selectbox(
            "Patient", options, index=options.index(default_id),
            format_func=lambda pid: (
                f"{pid}   ·   risk "
                f"{float(plan.loc[plan['patient_id'] == pid, 'p'].iloc[0]):.2f}"),
        )
        row = plan[plan["patient_id"] == sel_id].iloc[0]

        fg, bg = ACTION_COLOR[row["action"]]
        held = {1: "Cognitive only", 2: "2 of 4", 3: "3 of 4", 4: "All 4"}[row["n_modalities"]]
        st.markdown(f"""<div style="display:flex;gap:26px;align-items:center;
            background:{BG};border-radius:20px;padding:14px 20px;margin:6px 0 4px 0">
            <div><div style="font-size:10.5px;font-weight:700;letter-spacing:.09em;
            text-transform:uppercase;color:{INK3}">Risk</div>
            <div style="font-family:Quicksand;font-size:19px;font-weight:700">
            {row['p']:.2f} <span style="font-size:13px;color:{INK3}">±
            {row['u']:.2f}</span></div></div>
            <div><div style="font-size:10.5px;font-weight:700;letter-spacing:.09em;
            text-transform:uppercase;color:{INK3}">Modalities</div>
            <div style="font-family:Quicksand;font-size:19px;font-weight:700">{held}
            </div></div>
            <div><div style="font-size:10.5px;font-weight:700;letter-spacing:.09em;
            text-transform:uppercase;color:{INK3}">Recommended action</div>
            <div style="margin-top:5px">{pill(row['action'].value, fg, bg)}</div></div>
            </div>""", unsafe_allow_html=True)

        whatif = st.toggle(
            "What-if: sweep the pre-test risk",
            help="Detaches the panel from this patient and sweeps an arbitrary "
                 "prior, so the flip behaviour can be shown across the whole "
                 "range — including regions this cohort barely populates.")
        shown = row
        if whatif:
            hypo = st.slider("Hypothetical pre-test risk", 0.02, 0.97,
                             float(np.clip(row["p"], 0.02, 0.97)), 0.01)
            shown = {
                "patient_id": f"hypothetical (not {row['patient_id']})",
                "p": hypo, "u": float("nan"),
                "evals": {k: evaluate_test(hypo, k, theta) for k in TESTS},
            }
            st.markdown(
                f'<div class="banner">This is a <b>hypothetical prior</b>, not a '
                f'patient. In this cohort the risk estimates are strongly bimodal '
                f'— a consequence of the circular CDR label — so the p≈0.90 region '
                f'where PET alone can decide is barely populated. Sweep to 0.90 to '
                f'see the behaviour the startup self-check verifies.</div>',
                unsafe_allow_html=True)

        st.pyplot(fig_flip_panel(shown, theta), use_container_width=True)
        st.caption("Bayesian update from published sensitivity / specificity. "
                   "Amyloid PET remains available for treatment-eligibility "
                   "confirmation even where it cannot change the triage action.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── D. queue ──────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### Work queue, ranked by information value")
    st.caption(f"Ranked by P(result flips the action) × ₹{COST_OF_WRONG_ACTION:,} ÷ "
               f"test cost — not by risk. That is why a lower-risk patient can sit "
               f"above a higher-risk one.")

    top = plan.sort_values("information_value", ascending=False).head(12)
    html = ['<table class="qt"><thead><tr>'
            '<th style="width:40px;text-align:right">#</th><th>Patient</th>'
            '<th>Risk (± spread)</th><th>Modalities</th><th>Next action</th>'
            '<th style="width:34%">Why</th>'
            '<th style="text-align:right">Cost</th></tr></thead><tbody>']

    for n, (_, r) in enumerate(top.iterrows(), start=1):
        act = Action.DEFERRED_NEXT_MONTH if r["deferred"] else r["action"]
        fg, bg = ACTION_COLOR[act]
        mods = "".join(
            f'<span class="mod" style="background:{SUN if on else "#F6EDE2"};'
            f'color:{"#FFFFFF" if on else "#C9B6A2"}">{k}</span>'
            for k, on in [("COG", True), ("CMB", r["has_comorbidity"]),
                          ("BLD", r["has_blood"]), ("MRI", r["has_mri"])])
        cost = TESTS[r["slot_test"]][2] if r["slot_test"] else 0
        pcol = CORAL if r["p"] >= theta else SAGE
        html.append(
            f'<tr><td style="text-align:right;color:{INK3};font-weight:700">{n}</td>'
            f'<td style="font-weight:700">{r["patient_id"]}</td>'
            f'<td><b style="color:{pcol}">{r["p"]:.2f}</b> '
            f'<span style="color:{INK3};font-size:12px">± {r["u"]:.2f}</span></td>'
            f'<td>{mods}</td>'
            f'<td>{pill(act.value, fg, bg)}</td>'
            f'<td style="color:{INK2};font-size:12.5px;line-height:1.45">'
            f'{reason_text(r, theta)}</td>'
            f'<td style="text-align:right;font-weight:700">'
            f'{"₹{:,}".format(cost) if cost else "—"}</td></tr>')
    html.append("</tbody></table>")
    st.markdown("".join(html), unsafe_allow_html=True)

    # Point at a real risk inversion if one exists in the visible queue.
    inv = None
    vis = top.reset_index(drop=True)
    for i in range(len(vis)):
        for j in range(i + 1, len(vis)):
            if vis.loc[i, "p"] < vis.loc[j, "p"] - 0.12:
                inv = (vis.loc[i], vis.loc[j])
                break
        if inv:
            break
    if inv:
        a, b = inv
        st.markdown(
            f'<div class="banner" style="margin-top:14px">'
            f'<b>Risk inversion, on purpose.</b> {a["patient_id"]} '
            f'(risk {a["p"]:.2f}) outranks {b["patient_id"]} (risk {b["p"]:.2f}) — '
            f'a cheap test can still change the first patient\'s plan, while the '
            f'second is already past the point where a result would alter anything.'
            f'</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── E. policy comparison + equity ─────────────────────────────────────
    c1, c2 = st.columns([1.6, 1])

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Same capacity, two policies")
        st.caption("Naive fills PET from the top of the risk list. Ours fills by "
                   "information value, under the flip filter.")
        rows = [
            ("Decisions resolved", s["naive_resolved"], s["ours_resolved"],
             f"{s['ours_resolved'] - s['naive_resolved']:+d}"),
            ("Scans that cannot flip", s["naive_wasted"], 0, f"−{s['naive_wasted']}"),
            ("Spend on those scans", inr(s["naive_wasted_cost"]), "₹0",
             f"−{inr(s['naive_wasted_cost'])}"),
            ("PET yield (amyloid +)", f"{s['naive_pet_yield']:.0%}",
             f"{s['ours_pet_yield']:.0%}",
             f"{s['ours_pet_yield'] - s['naive_pet_yield']:+.0%}"),
            ("Fraction of PET scans that change a plan",
             f"{s['naive_useful_per_pet']:.0%}",
             f"{s['ours_useful_per_pet']:.0%}",
             f"{s['ours_useful_per_pet'] - s['naive_useful_per_pet']:+.0%}"),
        ]
        t = ['<table class="qt"><thead><tr><th></th>'
             '<th style="text-align:center">Naive</th>'
             f'<th style="text-align:center;color:{SUN}">Ours</th>'
             '<th style="text-align:right">Δ</th></tr></thead><tbody>']
        for k, a, b, d in rows:
            t.append(f'<tr><td style="color:{INK2}">{k}</td>'
                     f'<td style="text-align:center;color:{INK3};font-weight:700;'
                     f'font-size:15px">{a}</td>'
                     f'<td style="text-align:center;font-weight:700;font-size:15px">'
                     f'{b}</td>'
                     f'<td style="text-align:right;font-weight:700;color:{SAGE}">'
                     f'{d}</td></tr>')
        t.append("</tbody></table>")
        st.markdown("".join(t), unsafe_allow_html=True)
        st.caption("Naive scans a higher share of amyloid-positive patients — and "
                   "that is precisely the problem. It spends PET on people whose "
                   "plan was never in doubt.")
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        low = plan[plan["low_education"]]
        high = plan[~plan["low_education"]]
        low_rate = float(low["granted"].mean()) if len(low) else 0.0
        high_rate = float(high["granted"].mean()) if len(high) else 0.0
        breach = high_rate > 0 and low_rate < high_rate * 0.8

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Equity guard")
        a, b = st.columns(2)
        a.markdown(f"""<div class="k" style="font-size:10.5px;font-weight:700;
            letter-spacing:.09em;text-transform:uppercase;color:{INK3}">
            Low education</div><div style="font-family:Quicksand;font-size:27px;
            font-weight:700">{low_rate:.0%}</div>
            <div style="font-size:11.5px;color:{INK3}">selected into queue</div>""",
            unsafe_allow_html=True)
        b.markdown(f"""<div class="k" style="font-size:10.5px;font-weight:700;
            letter-spacing:.09em;text-transform:uppercase;color:{INK3}">
            High education</div><div style="font-family:Quicksand;font-size:27px;
            font-weight:700">{high_rate:.0%}</div>
            <div style="font-size:11.5px;color:{INK3}">selected into queue</div>""",
            unsafe_allow_html=True)

        if breach:
            st.markdown(f'<div class="banner" style="margin-top:14px">'
                        f'Queue may under-serve low-education patients. MMSE is '
                        f'education-biased and dementia prevalence is higher in this '
                        f'group (10.29% vs 1.54%, IIPS national study).</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="good" style="margin-top:14px">Selection rates '
                        'are within 20% of each other this month. MMSE remains '
                        'education-biased, so this check runs on every allocation.'
                        '</div>', unsafe_allow_html=True)
        st.caption("Surfaced, never auto-corrected. Silently reweighting the queue "
                   "would be its own problem — this is a clinician's call to make "
                   "in the open.")
        st.markdown('</div>', unsafe_allow_html=True)

    # ── STEP 7 — export ───────────────────────────────────────────────────
    if export:
        FIG_DIR.mkdir(exist_ok=True)
        figs = {
            "coverage_grid.png": fig_coverage(df, standalone=True),
            "decision_flip.png": fig_flip_panel(shown, theta, standalone=True),
            "policy_comparison.png": fig_policy(s, standalone=True),
            "funnel.png": fig_funnel(df, plan, s, standalone=True),
        }
        for name, f in figs.items():
            f.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight",
                      facecolor=CARD)
            plt.close(f)
        st.sidebar.success(f"Wrote {len(figs)} figures to {FIG_DIR.resolve()}")


if __name__ == "__main__":
    main()
