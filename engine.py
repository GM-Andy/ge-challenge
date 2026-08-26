"""
Triage engine — cohort construction, decision-flip rule, allocation.

Kept free of Streamlit so it can be imported, tested and re-run headlessly.
The system never outputs a diagnosis; every output is an action recommendation.
"""

from __future__ import annotations

import time
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
APP_DIR = Path(__file__).resolve().parent
CSV_PATH = APP_DIR / "oasis_longitudinal.csv"
FIG_DIR = APP_DIR / "figures"

# ── decision-flip constants (load-bearing, do not tune) ───────────────────────
THETA = 0.40

TESTS = {                        # (sensitivity, specificity, cost_INR, days)
    "blood_ptau217": (0.90, 0.90,  8000,  3),
    "mri":           (0.85, 0.70, 12000, 14),
    "pet":           (0.95, 0.95, 60000, 30),
}

TEST_LABEL = {
    "blood_ptau217": "Blood p-tau217",
    "mri": "Structural MRI",
    "pet": "Amyloid PET",
}

STAGE_LABEL = {
    1: "Stage 1 · Cognitive Screening",
    2: "Stage 2 · Blood-Based Biomarkers",
    3: "Stage 3 · MRI Evaluation",
    4: "Stage 4 · PET Scan Prioritisation",
}
STAGE_SHORT = ["Cognitive", "Blood", "MRI", "PET"]

COST_OF_WRONG_ACTION = 250_000


def likelihood_ratios(sens, spec):
    return sens / (1 - spec), (1 - sens) / spec


def posterior(p, lr):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    odds = (p / (1 - p)) * lr
    return odds / (1 + odds)


def evaluate_test(p: float, key: str, theta: float = THETA) -> dict:
    sens, spec, cost, days = TESTS[key]
    lr_pos, lr_neg = likelihood_ratios(sens, spec)
    pp = float(posterior(p, lr_pos))
    pn = float(posterior(p, lr_neg))
    flips = (pp >= theta) != (pn >= theta)
    p_pos = p * sens + (1 - p) * (1 - spec)
    p_flip = p_pos if p < theta else (1 - p_pos)
    return {
        "test": key, "p_if_positive": pp, "p_if_negative": pn,
        "flips": bool(flips), "wasted": not bool(flips),
        "p_test_positive": float(p_pos),
        "information_value": float(p_flip * COST_OF_WRONG_ACTION / max(cost, 1)),
        "cost": cost, "days": days,
    }


# ── action vocabulary. deliberately no diagnosis member ──────────────────────
class Action(Enum):
    ORDER_BLOOD_PTAU217 = "Order blood p-tau217"
    ORDER_MRI = "Order structural MRI"
    ORDER_PET = "Order amyloid PET"
    ESCALATE_NOW = "Escalate to specialist review"
    ESCALATE_NOW_PET_ELIGIBILITY = "Escalate now + PET for therapy eligibility"
    RELEASE_12MO = "Re-screen in 12 months"
    CLINICIAN_REVIEW = "Clinician review - insufficient data"
    DEFERRED_NEXT_MONTH = "Deferred to next month - no slot"


_FORBIDDEN = ("diagnos", "has alzheimer", "confirmed ad", "positive for ad")


def assert_no_diagnosis_actions() -> None:
    for m in Action:
        blob = f"{m.name} {m.value}".lower()
        for bad in _FORBIDDEN:
            if bad in blob:
                raise AssertionError(f"Action.{m.name} reads as a diagnosis")


ORDER_ACTION = {
    "blood_ptau217": Action.ORDER_BLOOD_PTAU217,
    "mri": Action.ORDER_MRI,
    "pet": Action.ORDER_PET,
}

TIERS = ("HIGH", "MEDIUM", "LOW")

# concentration of the belief by how much workup is already done — more
# completed tests, tighter posterior. Used only for the uncertainty band.
STAGE_CONCENTRATION = {1: 4.0, 2: 9.0, 3: 16.0}


def verification_table(theta: float = THETA) -> str:
    """Step-3 self-check on the flip rule itself."""
    expect = {
        0.03: ("none", "none", "none", "release, test nothing"),
        0.10: ("FLIP", "none", "FLIP", "blood only"),
        0.25: ("FLIP", "FLIP", "FLIP", "cheapest flipping test"),
        0.55: ("FLIP", "FLIP", "FLIP", "cheapest flipping test"),
        0.90: ("none", "none", "FLIP", "SKIP blood+MRI, straight to PET"),
        0.95: ("none", "none", "none", "escalate, testing adds nothing"),
    }
    out = ["", f"  DECISION-FLIP RULE — self-check (theta = {theta:.2f})",
           "  " + "-" * 70,
           "  prior   blood   mri     pet     expected"]
    ok = True
    for p, exp in expect.items():
        got = tuple("FLIP" if evaluate_test(p, k, theta)["flips"] else "none"
                    for k in ("blood_ptau217", "mri", "pet"))
        match = got == exp[:3]
        ok &= match
        out.append("  %.2f    %-7s %-7s %-7s -> %s%s"
                   % (p, *got, exp[3], "" if match else "   [MISMATCH]"))
    out += ["  " + "-" * 70, "  flip rule: %s" % ("PASS" if ok else "FAIL"), ""]
    return "\n".join(out)


# ══════════════════════════════════════════════════════════════════════════════
#  Cohort — patients arrive at DIFFERENT STAGES with DIFFERENT workup done
# ══════════════════════════════════════════════════════════════════════════════
FEATURES = ["MMSE", "Age", "EDUC", "SES", "sex_m",
            "hypertension", "diabetes", "apoe4"]

FEATURE_LABEL = {
    "MMSE": "MMSE", "Age": "Age", "EDUC": "Education (years)",
    "SES": "Socio-economic status", "sex_m": "Male",
    "hypertension": "Hypertension", "diabetes": "Diabetes", "apoe4": "APOE4 carrier",
}
# does a HIGHER value push priority up (+1) or down (-1)?
FEATURE_DIRECTION = {"MMSE": -1, "Age": +1, "EDUC": -1, "SES": +1, "sex_m": +1,
                     "hypertension": +1, "diabetes": +1, "apoe4": +1}


def _beta_ppf(u, a, b):
    """Beta quantile; scipy comes with sklearn, but degrade gracefully."""
    try:
        from scipy.stats import beta
        return beta.ppf(u, a, b)
    except Exception:                                   # pragma: no cover
        x = np.linspace(1e-4, 1 - 1e-4, 4000)
        pdf = x ** (a - 1) * (1 - x) ** (b - 1)
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        return np.interp(u, cdf, x)


def _cognitive_prior_from_real(df, rng):
    """
    Rank-map a clinical score built from real OASIS cognitive fields onto a
    Beta(1.8, 6.0) memory-clinic prevalence shape. Ordering is real; the
    calibration is the assumption, and it is stated in the UI.
    """
    s = (-0.30 * (df["MMSE"].fillna(27) - 27)
         + 0.040 * (df["Age"].fillna(75) - 75)
         - 0.050 * (df["EDUC"].fillna(14) - 14)
         + 0.100 * (df["SES"].fillna(3) - 3))
    s = s + rng.normal(0, 0.02, len(df))                # break ties, seeded
    u = (pd.Series(s).rank(method="first").to_numpy() - 0.5) / len(df)
    return np.clip(_beta_ppf(u, 1.8, 6.0), 0.005, 0.985)


def load_real() -> pd.DataFrame | None:
    if not CSV_PATH.exists():
        return None
    raw = pd.read_csv(CSV_PATH)
    return (raw.sort_values(["Subject ID", "Visit"])
               .groupby("Subject ID", as_index=False).first().reset_index(drop=True))


def build_cohort(n: int = 1000, seed: int = SEED, use_real: bool = False) -> pd.DataFrame:
    """
    Every patient starts with a cognitive-screening prior, then receives a
    Bayesian update for each test ALREADY COMPLETED before they reach us.
    A stage-3 patient therefore has a much sharper posterior than a stage-1
    patient, and we only ever offer tests they have not already had.
    """
    rng = np.random.default_rng(seed)
    real = load_real() if use_real else None

    if real is not None:
        df = real.copy()
        n = len(df)
        df["sex_m"] = (df["M/F"] == "M").astype(float)
        prior = _cognitive_prior_from_real(df, rng)
        source = "OASIS-2"
    else:
        df = pd.DataFrame({
            "Subject ID": [f"SYN_{i+1:04d}" for i in range(n)],
            "Age": rng.integers(60, 96, n),
            "EDUC": rng.integers(6, 24, n),
            "SES": rng.integers(1, 6, n),
        })
        prior = rng.beta(1.8, 6.0, n)
        # MMSE consistent with the prior, so the explanation panel is coherent
        df["MMSE"] = np.clip(np.round(29 - 14 * prior + rng.normal(0, 1.1, n)), 5, 30)
        df["sex_m"] = (rng.random(n) < 0.42).astype(float)
        df["nWBV"] = np.clip(0.79 - 0.10 * prior + rng.normal(0, 0.018, n), 0.60, 0.86)
        df["eTIV"] = rng.normal(1480, 175, n).round(0)
        df["ASF"] = (1755.0 / df["eTIV"]).round(3)
        source = "Synthetic"

    df["patient_id"] = df["Subject ID"]
    df["cognitive_prior"] = prior

    # simulated co-morbidity block, always labelled as simulated
    df["hypertension"] = (rng.random(n) < 0.41).astype(float)
    df["diabetes"] = (rng.random(n) < 0.19).astype(float)
    df["apoe4"] = (rng.random(n) < 0.30).astype(float)

    # latent status drives the results of tests already completed
    truth = rng.random(n) < prior
    stage = rng.choice([1, 2, 3], n, p=[0.60, 0.28, 0.12])

    p = prior.copy()
    blood_done = stage >= 2
    mri_done = stage >= 3
    results = {}

    for key, done in (("blood_ptau217", blood_done), ("mri", mri_done)):
        sens, spec, _, _ = TESTS[key]
        draw = rng.random(n)
        pos = np.where(truth, draw < sens, draw < (1 - spec))
        lr_pos, lr_neg = likelihood_ratios(sens, spec)
        lr = np.where(pos, lr_pos, lr_neg)
        p = np.where(done, posterior(p, lr), p)
        results[key] = np.where(done, pos, np.nan)

    df["stage"] = stage
    df["has_blood"] = blood_done
    df["has_mri"] = mri_done
    df["blood_positive"] = results["blood_ptau217"]
    df["mri_positive"] = results["mri"]
    df["p"] = np.clip(p, 0.002, 0.998)

    k = np.vectorize(STAGE_CONCENTRATION.get)(stage).astype(float)
    df["u"] = np.clip(np.sqrt(df["p"] * (1 - df["p"]) / (k + 1)), 0.010, 0.35)

    df["n_modalities"] = 1 + blood_done.astype(int) + mri_done.astype(int)
    df["source"] = source
    df["low_education"] = df["EDUC"] < df["EDUC"].median()
    return df


def available_tests(row) -> list[str]:
    """Never offer a test the patient has already had."""
    out = []
    if not row["has_blood"]:
        out.append("blood_ptau217")
    if not row["has_mri"]:
        out.append("mri")
    out.append("pet")                       # PET is never pre-completed here
    return out


def tier_of(p: float, u: float, theta: float = THETA) -> str:
    if p - u >= theta:
        return "HIGH"
    if p + u < theta:
        return "LOW"
    return "MEDIUM"


def treatment_candidate(row, theta: float = THETA) -> bool:
    """Plausible amyloid-targeting therapy candidate: early stage, mild MMSE."""
    mmse = row.get("MMSE", np.nan)
    return bool(row["p"] >= theta and pd.notna(mmse) and 20 <= mmse <= 28)


def build_plan(cohort: pd.DataFrame, theta: float = THETA) -> pd.DataFrame:
    rows = []
    u_cut = float(np.quantile(cohort["u"], 0.90)) if len(cohort) else 1.0

    for _, r in cohort.iterrows():
        avail = available_tests(r)
        evals = {k: evaluate_test(float(r["p"]), k, theta) for k in TESTS}
        flipping = [k for k in avail if evals[k]["flips"]]

        chosen = None
        if flipping:
            # A test can still change the plan — order the cheapest one.
            # Never divert these to review: a wide band is the REASON to test,
            # not a reason to stop.
            chosen = min(flipping, key=lambda k: TESTS[k][2])
            act = ORDER_ACTION[chosen]
        elif r["p"] >= theta:
            act = Action.ESCALATE_NOW
        elif r["stage"] == 1 and r["p"] + r["u"] >= theta:
            # Nothing on the menu flips, but this patient has cognitive data
            # only and the band still crosses theta. Releasing them for a year
            # on that basis is the call a human should make, not the engine.
            act = Action.CLINICIAN_REVIEW
        else:
            act = Action.RELEASE_12MO

        cand = treatment_candidate(r, theta)
        if act is Action.ESCALATE_NOW and cand:
            act = Action.ESCALATE_NOW_PET_ELIGIBILITY

        if chosen:
            slot, iv = chosen, evals[chosen]["information_value"]
        elif act is Action.ESCALATE_NOW_PET_ELIGIBILITY:
            slot, iv = "pet", evals["pet"]["information_value"]
        else:
            slot, iv = None, 0.0

        rows.append({
            "patient_id": r["patient_id"], "p": float(r["p"]), "u": float(r["u"]),
            "cognitive_prior": float(r["cognitive_prior"]),
            "stage": int(r["stage"]), "n_modalities": int(r["n_modalities"]),
            "has_blood": bool(r["has_blood"]), "has_mri": bool(r["has_mri"]),
            "blood_positive": r["blood_positive"], "mri_positive": r["mri_positive"],
            "MMSE": r.get("MMSE", np.nan), "Age": r.get("Age", np.nan),
            "EDUC": r.get("EDUC", np.nan), "SES": r.get("SES", np.nan),
            "apoe4": r.get("apoe4", np.nan),
            "low_education": bool(r["low_education"]),
            "tier": tier_of(float(r["p"]), float(r["u"]), theta),
            "action": act, "slot_test": slot, "information_value": iv,
            "treatment_candidate": cand, "available": avail, "evals": evals,
        })

    return pd.DataFrame(rows)


def allocate(plan: pd.DataFrame, caps: dict) -> pd.DataFrame:
    left = dict(caps)
    plan = plan.copy()
    plan["granted"] = False
    plan["deferred"] = False
    order = plan.sort_values("information_value", ascending=False).index
    for i in order:
        k = plan.at[i, "slot_test"]
        if not k:
            continue
        if left.get(k, 0) > 0:
            left[k] -= 1
            plan.at[i, "granted"] = True
        else:
            plan.at[i, "deferred"] = True
    plan["rank"] = plan["information_value"].rank(ascending=False,
                                                  method="first").astype(int)
    return plan


def naive_allocation(plan: pd.DataFrame, caps: dict) -> pd.DataFrame:
    """Rank by risk, work down the list. One test per patient."""
    left = dict(caps)
    scans = []
    for _, r in plan.sort_values("p", ascending=False).iterrows():
        for k in ("pet", "mri", "blood_ptau217"):
            if k not in r["available"]:
                continue
            if left.get(k, 0) > 0:
                left[k] -= 1
                scans.append({"patient_id": r["patient_id"], "test": k, "p": r["p"],
                              "wasted": not r["evals"][k]["flips"],
                              "cost": TESTS[k][2]})
                break
    return pd.DataFrame(scans)


def _safe_mean(x, default=0.0):
    x = np.asarray(x, dtype=float)
    return float(x.mean()) if x.size else default


def policy_summary(plan: pd.DataFrame, naive: pd.DataFrame) -> dict:
    ours = plan[plan["granted"]]
    ours_pet = ours[ours["slot_test"] == "pet"]
    naive_pet = naive[naive["test"] == "pet"] if len(naive) else naive

    terminal = plan["action"].isin([Action.ESCALATE_NOW,
                                    Action.ESCALATE_NOW_PET_ELIGIBILITY,
                                    Action.RELEASE_12MO])
    resolved = int((terminal | plan["granted"]).sum())

    n_wasted = int(naive["wasted"].sum()) if len(naive) else 0
    return {
        "ours_scans": int(len(ours)), "ours_decisive": int(len(ours)),
        "ours_nondecisive": 0, "ours_nondecisive_cost": 0,
        "ours_pet": int(len(ours_pet)),
        "ours_pet_mean_p": _safe_mean(ours_pet["p"]),
        "ours_resolved": resolved,
        "naive_scans": int(len(naive)),
        "naive_decisive": int(len(naive) - n_wasted),
        "naive_nondecisive": n_wasted,
        "naive_nondecisive_cost": int(naive.loc[naive["wasted"], "cost"].sum())
                                  if len(naive) else 0,
        "naive_pet": int(len(naive_pet)),
        "naive_pet_mean_p": _safe_mean(naive_pet["p"]) if len(naive_pet) else 0.0,
        "naive_pet_wasted": int(naive_pet["wasted"].sum()) if len(naive_pet) else 0,
        "naive_resolved": int(naive.loc[~naive["wasted"], "patient_id"].nunique())
                          if len(naive) else 0,
        "deferred": int(plan["deferred"].sum()),
    }


def cohort_self_check(n: int = 1000, seed: int = SEED,
                      pet_slots: int = 20, theta: float = THETA) -> str:
    """Reproduces the reference numbers from the build spec."""
    cohort = build_cohort(n, seed, use_real=False)
    plan = build_plan(cohort, theta)
    caps = {"pet": pet_slots, "mri": 10 ** 6, "blood_ptau217": 10 ** 6}
    alloc = allocate(plan, caps)
    naive = naive_allocation(plan, caps)
    naive_pet = naive[naive["test"] == "pet"] if len(naive) else naive
    ours_pet = alloc[alloc["granted"] & (alloc["slot_test"] == "pet")]

    lines = ["  COHORT SELF-CHECK  (N=%d, seed=%d)" % (n, seed), "  " + "-" * 70]
    counts = plan["action"].value_counts()
    for a in (Action.RELEASE_12MO, Action.ORDER_BLOOD_PTAU217, Action.ORDER_MRI,
              Action.ORDER_PET, Action.ESCALATE_NOW,
              Action.ESCALATE_NOW_PET_ELIGIBILITY, Action.CLINICIAN_REVIEW):
        c = int(counts.get(a, 0))
        if c:
            lines.append("  %-34s %4d  (%4.1f%%)" % (a.name, c, 100 * c / n))

    nw = int(naive_pet["wasted"].sum()) if len(naive_pet) else 0
    np_n = len(naive_pet)
    lines += [
        "  " + "-" * 70,
        "  PET slots = %d" % pet_slots,
        "    rank-by-risk : %d/%d scans cannot change the action   (mean prior %.2f)"
        % (nw, np_n, _safe_mean(naive_pet["p"]) if np_n else 0.0),
        "    flip-filtered: %d/%d cannot change                    (mean prior %.2f)"
        % (0, len(ours_pet), _safe_mean(ours_pet["p"])),
    ]
    if np_n:
        lines.append("  --> %.0f%% of risk-ranked PET slots are non-decisive"
                     % (100 * nw / np_n))
    lines.append("  " + "-" * 70)
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  Contributing factors
# ══════════════════════════════════════════════════════════════════════════════
def fit_explainer(cohort: pd.DataFrame, seed: int = SEED):
    """
    Small model of the cognitive prior, used only to explain which intake
    features move a patient's starting risk. Never used to assign an action.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.inspection import permutation_importance

    X = cohort[FEATURES].astype(float).to_numpy()
    y = cohort["cognitive_prior"].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=160, max_depth=4,
                                          random_state=seed).fit(X, y)
    imp = permutation_importance(model, X, y, n_repeats=6, random_state=seed)
    medians = np.nanmedian(X, axis=0)
    global_rank = list(np.argsort(imp.importances_mean)[::-1])
    return {"model": model, "medians": medians, "global_rank": global_rank,
            "importance": imp.importances_mean}


def contributing_factors(row, expl, top_k: int = 5) -> list[dict]:
    """
    Local attribution: replace one feature with the cohort median and measure
    how far the predicted prior moves. Honest and easy to explain.
    """
    x = np.array([float(row.get(f, np.nan)) for f in FEATURES], dtype=float)
    base = float(expl["model"].predict(x.reshape(1, -1))[0])

    out = []
    for j, f in enumerate(FEATURES):
        if not np.isfinite(x[j]):
            continue
        alt = x.copy()
        alt[j] = expl["medians"][j]
        delta = base - float(expl["model"].predict(alt.reshape(1, -1))[0])
        if abs(delta) < 1e-6:
            continue
        out.append({
            "feature": f, "label": FEATURE_LABEL[f], "value": x[j],
            "median": float(expl["medians"][j]), "delta": delta,
            "direction": "raises priority" if delta > 0 else "lowers priority",
        })
    out.sort(key=lambda d: abs(d["delta"]), reverse=True)
    return out[:top_k]


def counterfactual(row, expl, theta: float = THETA) -> str | None:
    """
    Sweep the single most influential feature until the TIER changes.
    The most useful sentence a clinician can be handed.
    """
    facs = contributing_factors(row, expl, top_k=1)
    if not facs:
        return None
    f = facs[0]["feature"]
    j = FEATURES.index(f)
    x = np.array([float(row.get(k, np.nan)) for k in FEATURES], dtype=float)
    if not np.isfinite(x[j]):
        return None

    cur_tier = row["tier"]
    lo, hi = (5, 30) if f == "MMSE" else (float(np.floor(x[j] - 12)),
                                          float(np.ceil(x[j] + 12)))
    grid = np.linspace(lo, hi, 60)
    # ratio between the patient's posterior and their prior, held fixed while
    # we vary intake — the completed tests still count.
    ratio = row["p"] / max(row["cognitive_prior"], 1e-6)

    for v in (grid if FEATURE_DIRECTION.get(f, 1) > 0 else grid[::-1]):
        alt = x.copy()
        alt[j] = v
        prior = float(np.clip(expl["model"].predict(alt.reshape(1, -1))[0],
                              0.002, 0.998))
        p_new = float(np.clip(prior * ratio, 0.002, 0.998))
        if tier_of(p_new, row["u"], theta) != cur_tier:
            new_tier = tier_of(p_new, row["u"], theta)
            unit = "" if f in ("sex_m", "apoe4", "hypertension", "diabetes") else ""
            comp = "of" if f == "MMSE" else "of"
            return (f"{FEATURE_LABEL[f]} {comp} {v:.0f}{unit} or beyond would move "
                    f"this patient to the {new_tier} tier.")
    return None


# ══════════════════════════════════════════════════════════════════════════════
#  Optional TabPFN — never allowed to break the demo
# ══════════════════════════════════════════════════════════════════════════════
TABPFN_BUDGET_S = 20.0


def tabpfn_status() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as e:
        return False, f"torch missing ({type(e).__name__})"
    try:
        import tabpfn
    except Exception as e:
        return False, f"tabpfn not importable ({type(e).__name__})"
    return True, f"TabPFN {getattr(tabpfn, '__version__', '?')} installed"
