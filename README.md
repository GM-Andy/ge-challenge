# Triage Engine for Alzheimer's Diagnostic Pathways

A clinical decision-support prototype for the **GE HealthCare Precision Care Challenge 2026**.

The system **never outputs a diagnosis**. Every output is an *action recommendation* — which test to order next, or release to monitoring.

```bash
streamlit run app.py
```

Requires only: `pip install streamlit pandas numpy scikit-learn matplotlib`

---

## The argument in one paragraph

Diagnostic capacity is the scarce resource, not diagnostic accuracy. A queue sorted by risk spends its PET slots on patients whose plan was never in doubt. This engine orders a test only when the test's result could land on the *other* side of the escalation threshold — when it can actually change what happens next. Everything else is spend and delay bought for nothing.

## The core idea — the decision-flip rule

For each patient and each test, update the risk under both possible results:

```
LR+ = sens / (1 - spec)          LR- = (1 - sens) / spec
posterior = (odds × LR) / (1 + odds × LR)
```

**A test is worth ordering only if `p_if_positive` and `p_if_negative` land on opposite sides of θ = 0.40.** If both land on the same side, the result cannot change the action, so the test is marked `wasted`.

The engine prints this self-check to the console at startup, and it **passes on the real data**:

```
prior   blood    mri      pet      resulting action
0.03    none     none     none     release, test nothing
0.10    FLIP     none     FLIP     blood only
0.25    FLIP     FLIP     FLIP     cheapest flipping test
0.55    FLIP     FLIP     FLIP     cheapest flipping test
0.90    none     none     FLIP     SKIP blood+MRI, straight to PET
0.95    none     none     none     escalate, testing adds nothing
```

The **0.90 row** is the point: a blood test on a 90% patient cannot change the plan, so ordering one wastes ₹8,000 and three days.

## V2 — staged cohort, tiers, stages, explanations

The engine now lives in `engine.py` (importable, no Streamlit) and the UI in `app.py`.

**The cohort was rebuilt, and this is the change that makes the demo work.** Previously one risk distribution was drawn for everyone, so almost all patients landed in the same action bucket. Now patients **arrive at different stages with different workup already done**:

```
stage 1 (60%)  cognitive screening only
stage 2 (28%)  cognitive + blood biomarker already resulted
stage 3 (12%)  cognitive + blood + MRI already resulted
```

Every patient starts with a cognitive prior, then receives a likelihood-ratio update for each test **already completed** — so a stage-3 patient has a much sharper posterior than a stage-1 patient. Only tests the patient has **not** already had are offered.

Startup self-check, N=1000, seed 42, against the reference numbers in the build spec:

| Action | Spec | This build |
|---|---|---|
| `RELEASE_12MO` | 208 (20.8%) | 219 (21.9%) |
| `ORDER_BLOOD_PTAU217` | 551 (55.1%) | 503 (50.3%) |
| `ORDER_MRI` | 42 (4.2%) | 49 (4.9%) |
| `ORDER_PET` | 190 (19.0%) | 204 (20.4%) |
| `ESCALATE_NOW` | 9 (0.9%) | 3 + 22 eligibility (2.5%) |

Every row lands within ~5 points. The escalate row splits because V2 adds the therapy-eligibility action, which the reference table predates.

### The "45% of PET slots are wasted" figure does not reproduce as a constant

The spec reports 9/20 (45%) of risk-ranked PET slots being non-decisive. Measured here it is **capacity- and cohort-dependent, not a fixed number**:

| Cohort | PET slots | Non-decisive, risk-ranked | Mean posterior: risk-ranked vs flip-filtered |
|---|---|---|---|
| Spec reference | 20 | 9 / 20 (45%) | 0.92 vs 0.50 |
| Synthetic, N=1000 | 20 | **20 / 20 (100%)** | **0.95 vs 0.61** |
| OASIS-2, N=150 | 20 | **1 / 20 (5%)** | **0.65 vs 0.34** |

The mean-posterior gap is the robust part: risk-ranking always sends the scanner the patients whose posterior is already highest, flip-filtering always sends a materially lower-posterior group, and the gap is wide in both cohorts (0.34 and 0.31 against the spec's 0.42).

The reason is arithmetic. PET stops being decisive above p ≈ 0.927. The synthetic cohort has 25 patients above that line, so any 20 slots filled from the top of the risk list are *all* non-decisive. The 150-patient OASIS cohort has barely any, so almost none are. Push the synthetic cohort to 60 PET slots and it lands near 42%, close to the spec's number.

The claim that survives is the directional one, and it holds in every configuration: **flip-filtering sends zero non-decisive PET scans, by construction.** The app reports the live percentage for whatever cohort and capacity is loaded rather than hard-coding 45%.

### What V2 adds

- **Priority tiers** — HIGH (`p − u ≥ θ`), LOW (`p + u < θ`), MEDIUM (band straddles θ), shown as a chip on every row with the cohort split in the header.
- **Stage tracker** — a 4-segment indicator per patient (Cognitive → Blood → MRI → PET), plus a cohort funnel by stage.
- **Treatment-eligibility flag** — MMSE 20–28 with a high posterior. PET stays recommended for these patients even where it cannot change the triage action, because eligibility confirmation is a separate clinical purpose. The reason is printed in the row rather than hidden.
- **Contributing factors** — top 5 drivers with direction and magnitude, in clinical language, plus a counterfactual line ("MMSE of 26 or above would move this patient to the LOW tier"). Global importance via `permutation_importance`; local attribution by substituting the cohort median for one feature and measuring how far the prediction moves.
- **Provenance panel** — per-modality source, real-vs-simulated status, the synthetic-surrogate identifier statement, and the MoCA-in-place-of-MMSE note.
- **Reference architecture** tab, exported as `architecture.png`.
- **Demo mode** — pins three real patients (hero / beneficiary / invisible) with plain-English captions. On by default; `?demo=0` turns it off.
- **Live deltas** — the capacity slider reports patients entering the plan and patients deferred, against the default capacity.

### Two places where the copy and the engine disagreed, and how it was settled

**Tier versus flip.** The spec's line is "Medium tier is where testing changes decisions. High and low are already decided." That is *mostly* true but not strictly: a LOW-tier stage-3 patient at p = 0.25 can still be top of the PET queue, because a positive PET moves them to 0.86 and a negative to 0.02 — the test genuinely decides. Tier answers "is today's estimate confidently one side of θ?"; the flip rule answers "could a result still cross it?". The app now says both, rather than letting the first queue row contradict the caption.

**Clinician review.** An earlier draft routed the top uncertainty decile to `CLINICIAN_REVIEW`. Because uncertainty peaks at p ≈ 0.5, that diverted exactly the patients a cheap blood test helps most, and it fired on a mechanical 10.0% of the cohort. It now fires only when no available test flips **and** the patient has cognitive data only **and** the band still crosses θ — the case where releasing someone for a year on thin data should be a human's call.

### …but that case does not occur in this cohort

Worth knowing before you demo it. Because the CDR label is circular (below), the model is strongly **bimodal** — 56 of 150 patients sit below 0.10 and 44 sit above 0.94:

```
0.00-0.10  56     0.40-0.60   9
0.10-0.25  21     0.60-0.85  12
0.25-0.40   6     0.85-0.94   2      <- the p≈0.90 window
                  0.94-1.00  44
```

The p≈0.90 region where PET alone can decide is nearly empty, so **zero** real patients here show the high-side money-shot. Eleven patients show its low-side mirror instead: risk ≈0.06, where blood and MRI cannot lift them over θ but PET could — a genuinely interesting "is ₹60,000 worth it?" case, and the panel's default selection.

To demonstrate the 0.90 behaviour, the flip panel has a **What-if: sweep the pre-test risk** toggle. It detaches the panel from any patient and sweeps an arbitrary prior, clearly banner-labelled as hypothetical. The behaviour it shows is the same one the startup self-check verifies.

## Data

Uses **OASIS-2** (`oasis_longitudinal.csv`, Kaggle `jboysen/mri-and-alzheimers`), collapsed to the first visit per subject — **150 patients**. If the CSV is absent the app generates a 400-patient synthetic cohort with identical column names and shows an amber banner saying so. It runs either way.

> **The CSV is not committed to this repo.** OASIS data carries a use agreement, and this repository is public, so the file is gitignored rather than redistributed. Download it from [Kaggle](https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers) and drop `oasis_longitudinal.csv` next to `app.py` — the app resolves it beside the script, not against your working directory. Without it you get the clearly-labelled synthetic cohort.
>
> OASIS-2 is courtesy of Marcus et al., and should be cited as: Marcus DS, Fotenos AF, Csernansky JG, Morris JC, Buckner RL. *Open Access Series of Imaging Studies (OASIS): Longitudinal MRI Data in Nondemented and Demented Older Adults.* Journal of Cognitive Neuroscience, 2010.

Four modality blocks, with deliberate missingness:

| Modality | Fields | Availability | Source |
|---|---|---|---|
| `cognitive` | MMSE, Age, EDUC, SES, M/F | 100% | **real** |
| `comorbidity` | hypertension, diabetes, APOE4 | ~60% | **simulated** (seed 42) |
| `blood` | ptau217_value | ~20% | **simulated** (seed 42) |
| `mri` | nWBV, eTIV, ASF | ~35% | **real**, randomly masked |

The missingness is the point, not a defect. A model requiring complete rows would score only the handful already fully worked up — and those are precisely the patients a clinician has already made up their mind about.

## Honest caveats

**The training label is partly circular.** `CDR > 0` is used as the progression proxy, but CDR is itself a clinician's rating made alongside the MMSE the model reads as a feature. The model is substantially learning to reproduce a judgement that has already been made. Prospective validation would use **conversion to AD over follow-up**, not a concurrent scale. Nothing here should be read as evidence of predictive accuracy.

**Sensitivity, specificity and cost figures** are published-literature estimates, not measured on this cohort. The cost of a wrong action (₹2,50,000) is an assumption, exposed in the sidebar so it can be argued with.

**Uncertainty is decomposed, and the headline correlation is weaker than you might expect.** Spread has two sources:

- `var_model` — bootstrap disagreement (15 models on resampled data) on the row *as it actually is*, NaNs included.
- `var_missing` — how far the estimate travels across plausible hot-deck completions of the modality blocks the patient does not hold. Zero by construction for a complete record.

Measured on the real cohort:

```
corr(n_modalities, total spread)            = -0.06  (Pearson)
                                            = -0.13  (Spearman)
corr(n_modalities, missing-data component)  = -0.44
mean spread by modality count:
  1 modality  0.150 (n=26)    3 modalities 0.105 (n=49)
  2 modalities 0.113 (n=70)   4 modalities 0.185 (n=5)
```

Fewer modalities do give wider spread — but only monotonically from 1→3. The five patients holding **all four** modalities have the *highest* spread of any group, because bootstrap variance peaks for genuinely borderline patients regardless of how well worked up they are. That confound drags the headline Pearson correlation to near zero. The claim that survives scrutiny is the **−0.44 on the missing-data component**, which is the quantity that actually means "we haven't measured enough".

Consequently `CLINICIAN_REVIEW` triggers on the **missing-data component**, not on total spread — its label reads *insufficient data*, and a fully worked-up borderline patient is not a data problem. In the current run all 15 patients routed to review hold only one or two modalities, which is the behaviour you want.

## Safety guard

If `p >= θ` and the patient is a candidate for amyloid-targeting therapy (`p >= θ`, `MMSE >= 20`, `age <= 85`), PET confirmation is **still required for treatment eligibility** even where it does not change the triage action. The engine labels that case `ESCALATE_NOW → PET for eligibility confirmation` and reserves the slot. It never suppresses a confirmatory PET for a treatment candidate.

The candidate rule is our own construction — it is not specified by any guideline reproduced here, and a clinician should set it.

## Ranking and capacity

```
information_value = P(result flips the action) × cost_of_wrong_action / test_cost
P(test positive)  = p·sens + (1-p)·(1-spec)
```

Slots are filled greedily by descending information value under the flip filter. Patients who do not fit roll to next month with a visible **deferred** state. Because ranking is by information value rather than risk, a lower-risk patient can legitimately outrank a higher-risk one — the queue's *Why* column explains each case, and the app flags a real inversion when one appears.

**The naive baseline gives each patient one test**, working down the risk list: PET to the highest-risk patients until those slots are gone, then MRI to the next tranche, then blood. Handing the same top-risk patient a PET *and* an MRI *and* a blood draw would make the comparison a strawman — the waste count would balloon simply because the policy triple-books one person.

At defaults (PET 20 / MRI 60 / blood 200, θ = 0.40) on the real cohort:

| | Naive | Ours |
|---|---|---|
| Decisions resolved | 52 | **135** / 150 |
| Scans that cannot flip | 98 | **0** |
| Spend on those scans | ₹19.4 L | **₹0** |
| PET yield (amyloid +) | **95%** | 46% |
| PET scans that change a plan | **0%** | **100%** |

Note the honest tension in the last two rows: the naive policy achieves a far *higher* PET yield, and that is exactly the indictment. It scans the patients most likely to be amyloid-positive — who are also the patients whose plan was never in doubt. Yield is the wrong metric; decisions changed per scan is the right one.

The 15 patients never resolved are the clinician-review cases. They are deliberately not auto-resolved.

## Equity guard

Patients are split at the median of `EDUC`. If the low-education selection rate falls more than 20% below the high-education rate, an amber warning appears:

> MMSE is education-biased and dementia prevalence is higher in this group (10.29% vs 1.54%, IIPS national study).

It is **surfaced, never auto-corrected**. Silently reweighting the queue would be its own problem; this is a clinician's call to make in the open.

## No diagnosis, enforced

The `Action` enum has no diagnosis member, and `assert_no_diagnosis_actions()` runs at startup — it raises if any action name or label reads as a diagnostic claim. The check is executed, not promised.

## Engine fallback

`HistGradientBoostingClassifier` is the default: it ingests NaN natively, so no row is imputed and no patient is dropped, and the whole 15-model bootstrap finishes in a couple of seconds.

**TabPFN is installed and works, but it is opt-in** via a sidebar checkbox. The reason is cost, measured on this cohort (150 rows × 12 features, CPU):

```
fit                 12.6 s
predict (150 rows)  17.5 s
predict (1200 rows) 48.3 s     <- the 8 imputation draws
```

Running the full 15-model bootstrap plus imputation through TabPFN would take roughly **20 minutes**, far past the 20-second budget. So when the box is ticked, **TabPFN supplies the point estimate only** (~15–30 s, cached) and the uncertainty bands stay with the bootstrap + imputation model. The sidebar says exactly that rather than implying the bands are TabPFN's.

Everything is wrapped in `try/except` with a timed budget: an import failure, a scikit-learn API break, a licence gate, or simply blowing 20 seconds all fall back silently to sklearn, and the sidebar names what actually ran.

Version notes worth knowing:

- **TabPFN 2.0.1 does not work with scikit-learn ≥ 1.6** — it calls `estimator._validate_data`, which sklearn removed. It raises `AttributeError` and the fallback catches it.
- **TabPFN ≥ 8.0 requires interactive licence acceptance** and a Prior Labs API key before it will download weights, so it cannot run unattended.
- This project therefore pins **`tabpfn==2.2.1`**, which installs cleanly and pulls weights from HuggingFace without a gate. Installing it **downgrades scikit-learn to 1.6.1**.

```bash
pip install streamlit pandas numpy scikit-learn matplotlib
pip install "tabpfn==2.2.1"   # optional; downgrades scikit-learn to 1.6.1
```

## Export for slides

**Export figures** in the sidebar writes four standalone-legible PNGs at 150 dpi to `./figures/`: `coverage_grid.png`, `decision_flip.png`, `policy_comparison.png`, `funnel.png`.

## Design

Companion design canvas (interactive, browser-based): the console plus two deck figures, built in the headspace-meditation register — warm cream, orange suns, soft rounded cards.
