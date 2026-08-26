"""
Triage Engine for Alzheimer's Diagnostic Pathways
GE HealthCare Precision Care Challenge 2026

Never outputs a diagnosis. Every output is an action recommendation.
Visual system: MindMarket (see DESIGN_mindmarket.md).

Run:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import engine as E
from engine import (Action, TESTS, TEST_LABEL, THETA, STAGE_SHORT, STAGE_LABEL,
                    SEED, FIG_DIR, COST_OF_WRONG_ACTION)

# ── MindMarket tokens ────────────────────────────────────────────────────────
CREAM = "#f5f1e4"
WHITE = "#ffffff"
INK = "#2c2e2a"
GRASS = "#8ed462"
SAND = "#e0dbce"
STONE = "#80827f"
MIST = "#d5d5d4"
SKY = "#2ba0ff"
CORAL = "#ff705d"
SUN = "#f5e211"

plt.rcParams.update({
    "font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14,
    "xtick.labelsize": 13, "ytick.labelsize": 13,
    "font.family": ["Inter", "DejaVu Sans"], "text.color": INK,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
})

TIER_STYLE = {                      # ink / grass / sand — no rainbow semantics
    "HIGH":   (WHITE, INK),
    "MEDIUM": (INK, GRASS),
    "LOW":    (STONE, SAND),
}
ACTION_DOT = {
    Action.ORDER_BLOOD_PTAU217: SKY,
    Action.ORDER_MRI: SKY,
    Action.ORDER_PET: CORAL,
    Action.ESCALATE_NOW: CORAL,
    Action.ESCALATE_NOW_PET_ELIGIBILITY: CORAL,
    Action.RELEASE_12MO: GRASS,
    Action.CLINICIAN_REVIEW: SUN,
    Action.DEFERRED_NEXT_MONTH: MIST,
}

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
.stApp {{ background: {CREAM}; }}
html, body, [class*="css"] {{ font-family: 'Inter', ui-sans-serif, system-ui, sans-serif;
  color: {INK}; }}
h1,h2,h3,h4 {{ font-family: 'Inter', sans-serif !important; font-weight: 500 !important;
  color: {INK} !important; }}
#MainMenu, footer, header {{ visibility: hidden; }}
.block-container {{ padding: 1.4rem 2.4rem 4rem 2.4rem; max-width: 1500px; }}
section[data-testid="stSidebar"] {{ background: {WHITE}; border-right: 1px solid {MIST}; }}

.hero h1 {{ font-size: 64px; font-weight: 500; letter-spacing: -3.8px;
  line-height: 0.98; margin: 0; }}
.hero p {{ font-size: 20px; line-height: 1.35; color: {INK}; margin: 14px 0 0 0;
  max-width: 760px; }}

.card {{ background: {WHITE}; border-radius: 50px; padding: 26px 30px;
  margin-bottom: 20px; }}
.card-tight {{ background: {WHITE}; border-radius: 34px; padding: 20px 22px; }}
.stat {{ background: {WHITE}; border-radius: 50px; padding: 22px 26px; }}
.stat .k {{ font-size: 15px; color: {STONE}; }}
.stat .v {{ font-size: 53px; font-weight: 500; letter-spacing: -2.12px;
  line-height: 1.05; margin-top: 4px; }}
.stat .s {{ font-size: 15px; color: {STONE}; margin-top: 6px; line-height: 1.4; }}

.chip {{ display:inline-block; padding:5px 14px; border-radius:50px;
  font-size:15px; font-weight:500; white-space:nowrap; }}
.tag {{ display:inline-block; padding:3px 11px; border-radius:10px;
  font-size:15px; border:1px solid {MIST}; color:{STONE}; }}
.qt {{ width:100%; border-collapse:collapse; }}
.qt th {{ font-size:15px; color:{STONE}; text-align:left; font-weight:400;
  padding:0 12px 12px 12px; }}
.qt td {{ padding:14px 12px; border-top:1px solid {MIST}; font-size:15px;
  vertical-align:middle; }}
.stButton>button {{ border-radius:50px; border:1px solid {INK}; background:{WHITE};
  color:{INK}; font-weight:500; padding:11px 22px; font-size:15px; }}
.stButton>button:hover {{ border-color:{GRASS}; color:{INK}; }}
.note {{ background:{SAND}; border-radius:34px; padding:18px 22px; font-size:15px;
  line-height:1.5; }}
.band {{ background:{SUN}; border-radius:34px; padding:18px 22px; font-size:15px;
  line-height:1.5; }}
</style>
"""


def chip(text, fg, bg):
    return f'<span class="chip" style="color:{fg};background:{bg}">{text}</span>'


def dot(color):
    return (f'<span style="display:inline-block;width:9px;height:9px;'
            f'border-radius:50px;background:{color};margin-right:8px;'
            f'vertical-align:middle"></span>')


def inr(v):
    v = float(v)
    if v >= 1e7:
        return f"₹{v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"₹{v/1e5:.1f} L"
    return f"₹{v:,.0f}"


def stage_track(stage: int, action) -> str:
    """Completed filled, current outlined, future empty."""
    cur = 3 if action in (Action.ORDER_PET, Action.ESCALATE_NOW,
                          Action.ESCALATE_NOW_PET_ELIGIBILITY) else stage
    out = []
    for i, name in enumerate(STAGE_SHORT):
        if i < stage:
            s = f"background:{GRASS};border:1px solid {GRASS}"
        elif i == cur:
            s = f"background:{WHITE};border:2px solid {INK}"
        else:
            s = f"background:{WHITE};border:1px solid {MIST}"
        out.append(f'<div title="{name}" style="width:26px;height:10px;'
                   f'border-radius:50px;{s}"></div>')
    return '<div style="display:flex;gap:5px;align-items:center">' + "".join(out) + "</div>"


# ══════════════════════════════════════════════════════════════════════════════
#  Figures
# ══════════════════════════════════════════════════════════════════════════════
def _clean(ax):
    ax.set_facecolor(WHITE)
    for s in ax.spines.values():
        s.set_visible(False)


def fig_flip(row, theta, standalone=False):
    fig, ax = plt.subplots(figsize=(11, 4.2), dpi=150)
    fig.patch.set_facecolor(WHITE)
    _clean(ax)
    keys = ["blood_ptau217", "mri", "pet"]
    halo = dict(facecolor=WHITE, edgecolor="none", pad=2.5)

    ax.axvline(theta, color=CORAL, lw=2.6, zorder=1)
    ax.text(theta, 2.75, f"θ = {theta:.2f}  escalate →", color=CORAL, fontsize=15,
            fontweight="bold", ha="center", bbox=halo, zorder=5)
    pl = float(np.clip(row["p"], 0.08, 0.92))
    ax.axvline(row["p"], color=INK, lw=1.8, ls=(0, (4, 3)), zorder=1)
    ax.text(pl, -0.85, f"risk now {row['p']:.2f}", color=INK, fontsize=14,
            fontweight="bold", ha="center", bbox=halo, zorder=5)

    for k, y in zip(keys, [2, 1, 0]):
        e = row["evals"][k]
        done = (k == "blood_ptau217" and row["has_blood"]) or \
               (k == "mri" and row["has_mri"])
        col = GRASS if (e["flips"] and not done) else MIST
        alpha = 1.0 if (e["flips"] and not done) else 0.55
        lo, hi = sorted((e["p_if_negative"], e["p_if_positive"]))
        ax.plot([lo, hi], [y, y], color=col, lw=10, alpha=alpha * .6,
                solid_capstyle="round", zorder=2)
        ax.scatter([e["p_if_negative"]], [y], s=240, facecolor=WHITE,
                   edgecolor=INK if not done else MIST, linewidth=3, zorder=3)
        ax.scatter([e["p_if_positive"]], [y], s=240,
                   color=INK if not done else MIST, zorder=3)
        ax.text(-0.03, y, f"{TEST_LABEL[k]}\n₹{TESTS[k][2]:,} · {TESTS[k][3]}d",
                ha="right", va="center", fontsize=14,
                color=INK if not done else STONE, linespacing=1.5)
        if done:
            stamp = "ALREADY DONE"
        elif e["flips"]:
            stamp = "THIS TEST DECIDES"
        else:
            stamp = "CANNOT CHANGE DECISION"
        ax.text(1.03, y, stamp, ha="left", va="center", fontsize=14,
                color=INK if e["flips"] and not done else STONE)

    ax.set_xlim(0, 1); ax.set_ylim(-1.1, 3.1)
    ax.set_yticks([]); ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xlabel("probability of progression", labelpad=10)
    if standalone:
        fig.suptitle("A test is worth ordering only if it can change the plan",
                     fontsize=18, fontweight="bold", y=1.06)
    fig.subplots_adjust(left=0.17, right=0.79, top=0.86, bottom=0.22)
    return fig


def fig_tiers(plan, standalone=False):
    counts = [int((plan["tier"] == t).sum()) for t in E.TIERS]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(WHITE); _clean(ax)
    cols = [INK, GRASS, SAND]
    bars = ax.bar(list(E.TIERS), counts, color=cols, width=.6, zorder=3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(counts) * .03,
                f"{c}", ha="center", fontsize=16, fontweight="bold")
    ax.set_yticks([]); ax.set_ylim(0, max(counts) * 1.2)
    ax.set_title("Priority tiers", fontsize=18, fontweight="bold", pad=16)
    ax.set_xlabel("Medium tier is where testing changes decisions.\n"
                  "High and low are already decided.", labelpad=14, fontsize=14)
    fig.tight_layout()
    return fig


def fig_funnel(plan, standalone=False):
    stages = [1, 2, 3]
    counts = [int((plan["stage"] == s).sum()) for s in stages]
    moving = int(plan["granted"].sum())
    labels = [STAGE_LABEL[s].replace(" · ", "\n") for s in stages]
    labels.append("Stage 4\nPET Prioritisation")
    counts.append(int(((plan["slot_test"] == "pet") & plan["granted"]).sum()))

    fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)
    fig.patch.set_facecolor(WHITE); _clean(ax)
    ys = np.arange(len(counts))[::-1]
    ax.barh(ys, counts, height=.55, color=[GRASS, SKY, CORAL, INK], zorder=3)
    for y, c in zip(ys, counts):
        ax.text(c + max(counts) * .015, y, f"{c}", va="center", fontsize=15,
                fontweight="bold")
    ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=13, linespacing=1.4)
    ax.set_xticks([]); ax.set_xlim(0, max(counts) * 1.15)
    ax.set_title(f"Cohort by stage — {moving} patients move stage this cycle",
                 fontsize=17, fontweight="bold", pad=16)
    fig.tight_layout()
    return fig


def fig_coverage(plan, n_show=40, standalone=False):
    sub = plan.head(n_show)
    fig, ax = plt.subplots(figsize=(6, 8.4), dpi=150)
    fig.patch.set_facecolor(WHITE); _clean(ax)
    for i, (_, r) in enumerate(sub.iterrows()):
        on = [True, bool(r["has_blood"]), bool(r["has_mri"]), False]
        for j, o in enumerate(on):
            ax.add_patch(plt.Rectangle((j + .08, i + .1), .84, .8,
                                       facecolor=GRASS if o else SAND,
                                       edgecolor="none", zorder=2))
    ax.set_xlim(0, 4); ax.set_ylim(len(sub), 0)
    ax.set_xticks([.5, 1.5, 2.5, 3.5]); ax.set_xticklabels(STAGE_SHORT, fontsize=14)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([]);
    done = int((plan["stage"] == 3).sum())
    ax.set_xlabel(f"{len(plan)-done} of {len(plan)} patients arrive with an\n"
                  f"incomplete workup. Nobody is dropped for it.",
                  labelpad=16, fontsize=14, linespacing=1.5)
    ax.xaxis.set_label_position("bottom")
    if standalone:
        fig.suptitle("Modality coverage at intake", fontsize=18,
                     fontweight="bold", y=.97)
    fig.tight_layout()
    return fig


def fig_policy(s, standalone=False):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4), dpi=150)
    fig.patch.set_facecolor(WHITE)
    panels = [
        ("Decisive scans", s["naive_decisive"], s["ours_decisive"], "{:,.0f}"),
        ("Non-decisive scans", s["naive_nondecisive"], 0, "{:,.0f}"),
        ("Spend on those", s["naive_nondecisive_cost"], 0, None),
        ("Mean posterior,\nPET group", s["naive_pet_mean_p"], s["ours_pet_mean_p"], "{:.2f}"),
    ]
    for ax, (t, a, b, fmt) in zip(axes, panels):
        _clean(ax)
        bars = ax.bar(["Risk-\nranked", "Flip-\nfiltered"], [a, b],
                      color=[SAND, GRASS], width=.6, zorder=3)
        for r, v in zip(bars, [a, b]):
            lab = inr(v) if fmt is None else fmt.format(v)
            ax.text(r.get_x() + r.get_width() / 2,
                    r.get_height() + max(a, b, 1e-9) * .05, lab,
                    ha="center", fontsize=14, fontweight="bold")
        ax.set_title(t, fontsize=15, fontweight="bold", pad=12)
        ax.set_ylim(0, max(a, b, 1e-9) * 1.3); ax.set_yticks([])
    if standalone:
        fig.suptitle("Same PET capacity, two ways to fill it", fontsize=18,
                     fontweight="bold", y=1.04)
    fig.tight_layout()
    return fig


def fig_factors(facs, standalone=False):
    fig, ax = plt.subplots(figsize=(8.4, 3.8), dpi=150)
    fig.patch.set_facecolor(WHITE); _clean(ax)
    if not facs:
        ax.text(.5, .5, "No contributing factors available", ha="center",
                fontsize=15, color=STONE); ax.axis("off"); return fig
    names = [f"{f['label']} {f['value']:.0f}" if f["value"] > 1.5
             else f"{f['label']}" for f in facs][::-1]
    vals = [f["delta"] for f in facs][::-1]
    ys = np.arange(len(vals))
    ax.barh(ys, vals, height=.55, color=[CORAL if v > 0 else GRASS for v in vals],
            zorder=3)
    ax.axvline(0, color=INK, lw=1.2)
    ax.set_yticks(ys); ax.set_yticklabels(names, fontsize=14)
    ax.set_xlabel("← lowers priority          raises priority →", labelpad=10)
    ax.set_xticks([])
    if standalone:
        fig.suptitle("Contributing factors", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def fig_architecture(standalone=True):
    built_row = ["Ingest\nscreening +\nclinical records",
                 "Standardise\n& link at\npatient level",
                 "Risk scoring\nwith\nuncertainty",
                 "Decision-flip\nfilter",
                 "Capacity-\nconstrained\nallocation",
                 "Clinician\ninterface"]
    design_row = ["EHR / PACS\nwrite-back",
                  "Prospective validation\n(conversion to AD)"]

    W, GAP, H = 1.80, 0.34, 0.40
    fig, ax = plt.subplots(figsize=(13, 5.0), dpi=150)
    fig.patch.set_facecolor(WHITE)
    ax.axis("off")

    y = 0.42
    for i, name in enumerate(built_row):
        x = i * (W + GAP)
        ax.add_patch(plt.Rectangle((x, y), W, H, facecolor=GRASS, edgecolor=INK,
                                   linewidth=1.4, zorder=3))
        ax.text(x + W / 2, y + H / 2, name, ha="center", va="center",
                fontsize=12, linespacing=1.4, color=INK, zorder=4)
        if i < len(built_row) - 1:
            ax.annotate("", xy=(x + W + GAP - 0.04, y + H / 2),
                        xytext=(x + W + 0.04, y + H / 2),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))

    for i, name in enumerate(design_row):
        x = i * (W + GAP) + 0.5 * (W + GAP)
        ax.add_patch(plt.Rectangle((x, 0.02), W, 0.30, facecolor=WHITE,
                                   edgecolor=STONE, linewidth=1.3,
                                   linestyle=(0, (4, 3)), zorder=3))
        ax.text(x + W / 2, 0.17, name, ha="center", va="center",
                fontsize=11, linespacing=1.4, color=STONE, zorder=4)

    total = len(built_row) * (W + GAP) - GAP
    ax.set_xlim(-0.15, total + 0.15)
    ax.set_ylim(-0.02, 1.06)

    ax.add_patch(plt.Rectangle((0, 0.94), 0.28, 0.07, facecolor=GRASS,
                               edgecolor=INK, lw=1.2))
    ax.text(0.38, 0.975, "Built in this prototype", fontsize=12.5,
            color=INK, va="center")
    ax.add_patch(plt.Rectangle((3.1, 0.94), 0.28, 0.07, facecolor=WHITE,
                               edgecolor=STONE, lw=1.2, linestyle=(0, (3, 2))))
    ax.text(3.48, 0.975, "Design stage", fontsize=12.5, color=STONE, va="center")

    fig.suptitle("Reference architecture", fontsize=18, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  Cached compute
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_cohort(n, seed, use_real):
    return E.build_cohort(n, seed, use_real)


@st.cache_data(show_spinner=False)
def get_plan(n, seed, use_real, theta):
    return E.build_plan(get_cohort(n, seed, use_real), theta)


@st.cache_resource(show_spinner=False)
def get_explainer(n, seed, use_real):
    return E.fit_explainer(get_cohort(n, seed, use_real), seed)


def pick_demo_cases(plan):
    """Real patients matching the three demo profiles. Returns {label: id}."""
    out = {}
    a = plan[(plan["p"] >= 0.90) &
             plan["evals"].map(lambda e: not e["pet"]["flips"])]
    if len(a):
        out["A"] = a.sort_values("p", ascending=False).iloc[0]["patient_id"]
    b = plan[(plan["p"].between(0.40, 0.62)) & (plan["slot_test"] == "pet")]
    if not len(b):
        b = plan[plan["slot_test"] == "pet"]
    if len(b):
        out["B"] = b.sort_values("information_value", ascending=False).iloc[0]["patient_id"]
    c = plan[(plan["stage"] == 1)].sort_values("u", ascending=False)
    if len(c):
        out["C"] = c.iloc[0]["patient_id"]
    return out


DEMO_CAPTION = {
    "A": "The hero. Ranked near the top by risk, but no scan can change the "
         "plan — escalate to specialist review and release the slot.",
    "B": "The beneficiary. Lower risk, but a PET here flips the decision "
         "cleanly. This patient takes the slot Case A released.",
    "C": "The invisible patient. Cognitive data only and a wide band — a model "
         "requiring complete rows would drop them entirely.",
}


# ══════════════════════════════════════════════════════════════════════════════
def main():
    E.assert_no_diagnosis_actions()
    print(E.verification_table())
    print(E.cohort_self_check())

    st.set_page_config(page_title="Triage Engine · Alzheimer's Pathways",
                       page_icon="🌱", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    qp = st.query_params
    demo_default = qp.get("demo", "1") not in ("0", "false", "no")

    with st.sidebar:
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:11px;margin-bottom:22px">'
            f'<div style="width:38px;height:38px;border-radius:10px;background:{GRASS}">'
            f'</div><div style="font-size:17px;font-weight:500;line-height:1.15">'
            f'Triage Engine<br><span style="font-size:15px;color:{STONE}">'
            f"Alzheimer's pathways</span></div></div>", unsafe_allow_html=True)

        demo_mode = st.toggle("Demo mode", value=demo_default,
                              help="Pins three patients that show the key behaviours.")
        cohort_src = st.radio("Cohort", ["OASIS-2 (real, 150)",
                                         "Synthetic memory clinic (1000)"],
                              index=1 if not E.CSV_PATH.exists() else 0)
        use_real = cohort_src.startswith("OASIS")
        n = 150 if use_real else 1000

        st.markdown("##### Monthly capacity")
        pet = st.slider("PET slots", 0, 100, 20)
        mri = st.slider("MRI slots", 0, 200, 60)
        blood = st.slider("Blood p-tau217", 0, 400, 200)
        theta = st.slider("Escalation threshold θ", 0.05, 0.95, THETA, 0.05)

        tab_ok, tab_probe = E.tabpfn_status()
        st.markdown(
            f'<div class="card-tight" style="margin-top:16px;background:{CREAM}">'
            f'<div style="font-size:15px;color:{STONE}">Risk engine</div>'
            f'<div style="font-size:17px;margin-top:4px">Staged Bayesian update</div>'
            f'<div style="font-size:15px;color:{STONE};margin-top:8px;line-height:1.45">'
            f'Cognitive prior, then a likelihood-ratio update for every test '
            f'already completed. {tab_probe}.</div></div>', unsafe_allow_html=True)

        export = st.button("Export figures → ./figures", use_container_width=True)

    caps = {"pet": pet, "mri": mri, "blood_ptau217": blood}
    plan = E.allocate(get_plan(n, SEED, use_real, theta), caps)
    naive = E.naive_allocation(plan, caps)
    s = E.policy_summary(plan, naive)

    # live delta vs the default capacity
    base = E.allocate(get_plan(n, SEED, use_real, theta),
                      {"pet": 20, "mri": 60, "blood_ptau217": 200})
    d_in = int(plan["granted"].sum() - base["granted"].sum())
    d_def = int(plan["deferred"].sum() - base["deferred"].sum())

    tab_main, tab_arch = st.tabs(["  Triage plan  ", "  Architecture & data  "])

    # ─────────────────────────── main tab ────────────────────────────────
    with tab_main:
        st.markdown(
            f'<div class="hero"><h1>Who gets the scan<br>this month?</h1>'
            f'<p>{len(plan)} patients, arriving at different stages with different '
            f'workup already done. Each one gets a next action — never a diagnosis.'
            f'</p></div>', unsafe_allow_html=True)
        st.write("")

        tiers = {t: int((plan["tier"] == t).sum()) for t in E.TIERS}
        cols = st.columns(4)
        cards = [
            ("HIGH tier", tiers["HIGH"], "already above θ — decided"),
            ("MEDIUM tier", tiers["MEDIUM"], "band straddles θ — testing pays here"),
            ("LOW tier", tiers["LOW"], "already below θ — decided"),
            ("PET slots used", f"{s['ours_pet']} / {pet}",
             f"{d_in:+d} in plan · {d_def:+d} deferred vs default"),
        ]
        for c, (k, v, sub) in zip(cols, cards):
            c.markdown(f'<div class="stat"><div class="k">{k}</div>'
                       f'<div class="v">{v}</div><div class="s">{sub}</div></div>',
                       unsafe_allow_html=True)
        st.markdown(
            f'<p style="font-size:18px;margin:18px 0 4px 0">Medium tier is where '
            f'testing changes decisions. High and low are already decided.</p>'
            f'<p style="font-size:15px;color:{STONE};margin:0 0 6px 0;max-width:900px">'
            f'Tier answers "is today\'s estimate confidently one side of θ?". The '
            f'flip rule answers "could a result still cross it?". They usually '
            f'agree — a low-tier patient is still offered a test when one '
            f'genuinely could move them, which is why a few appear in the queue.'
            f'</p>', unsafe_allow_html=True)

        # demo cases
        cases = pick_demo_cases(plan)
        sel_id = None
        if demo_mode and cases:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Demo cases")
            cc = st.columns(len(cases))
            keys = list(cases)
            if "demo_case" not in st.session_state:
                st.session_state.demo_case = keys[0]
            for col, k in zip(cc, keys):
                with col:
                    if st.button(f"Case {k} · {cases[k]}", key=f"btn{k}",
                                 use_container_width=True):
                        st.session_state.demo_case = k
                    st.markdown(f'<div style="font-size:15px;color:{STONE};'
                                f'line-height:1.45;margin-top:8px">'
                                f'{DEMO_CAPTION[k]}</div>', unsafe_allow_html=True)
            sel_id = cases.get(st.session_state.demo_case)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── selected patient ────────────────────────────────────────────
        opts = list(plan.sort_values("information_value", ascending=False)["patient_id"])
        if sel_id not in opts:
            sel_id = opts[0] if opts else None
        if sel_id is None:
            st.warning("No patients in this cohort."); return

        idx = opts.index(sel_id)
        pick = st.selectbox("Patient", opts, index=idx)
        row = plan[plan["patient_id"] == pick].iloc[0]

        left, right = st.columns([1.5, 1])
        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Can a test still change what we do?")
            fg, bg = TIER_STYLE[row["tier"]]
            act = Action.DEFERRED_NEXT_MONTH if row["deferred"] else row["action"]
            st.markdown(
                f'<div style="display:flex;gap:20px;align-items:center;flex-wrap:wrap;'
                f'margin-bottom:10px">{chip(row["tier"], fg, bg)}'
                f'<span style="font-size:20px">risk {row["p"]:.2f} '
                f'<span style="color:{STONE}">± {row["u"]:.2f}</span></span>'
                f'{stage_track(int(row["stage"]), row["action"])}'
                f'<span style="font-size:17px">{dot(ACTION_DOT[act])}{act.value}</span>'
                f'</div>', unsafe_allow_html=True)
            if row["treatment_candidate"]:
                st.markdown('<div class="band">Confirmatory PET indicated for '
                            'therapy eligibility. Eligibility confirmation is a '
                            'separate clinical purpose from triage, so the scan '
                            'stays recommended even where it cannot change the '
                            'triage action.</div>', unsafe_allow_html=True)
            st.pyplot(fig_flip(row, theta), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### What is driving this score")
            expl = get_explainer(n, SEED, use_real)
            facs = E.contributing_factors(row, expl)
            st.pyplot(fig_factors(facs), use_container_width=True)
            for f in facs:
                st.markdown(
                    f'<div style="font-size:15px;margin-bottom:5px">'
                    f'{f["label"]} <b>{f["value"]:.0f}</b> '
                    f'<span style="color:{STONE}">(cohort median '
                    f'{f["median"]:.0f})</span> → {f["direction"]}</div>',
                    unsafe_allow_html=True)
            cf = E.counterfactual(row, expl, theta)
            if cf:
                st.markdown(f'<div class="note" style="margin-top:14px">{cf}</div>',
                            unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── queue ────────────────────────────────────────────────────────
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Work queue, ranked by information value")
        st.caption(f"P(result flips the action) × {inr(COST_OF_WRONG_ACTION)} ÷ test "
                   f"cost — not by risk. A lower-risk patient can outrank a "
                   f"higher-risk one.")
        # Blood is the cheapest test, so it dominates information value and a
        # single combined list is 12 identical rows. Queue per resource, and
        # default to PET — the scarce one, where the ordering actually bites.
        qsel = st.radio("Queue", ["Amyloid PET", "Structural MRI",
                                  "Blood p-tau217", "All resources"],
                        index=0, horizontal=True, label_visibility="collapsed")
        key = {"Amyloid PET": "pet", "Structural MRI": "mri",
               "Blood p-tau217": "blood_ptau217"}.get(qsel)
        pool = plan if key is None else plan[plan["slot_test"] == key]
        if not len(pool):
            st.info(f"No patients are queued for {qsel} at this threshold.")
            pool = plan
        top = pool.sort_values("information_value", ascending=False).head(12)
        html = ['<table class="qt"><thead><tr><th>#</th><th>Patient</th><th>Tier</th>'
                '<th>Stage</th><th>Risk</th><th>Next action</th><th style="width:30%">'
                'Why</th><th style="text-align:right">Cost</th></tr></thead><tbody>']
        for i, (_, r) in enumerate(top.iterrows(), 1):
            a = Action.DEFERRED_NEXT_MONTH if r["deferred"] else r["action"]
            fg, bg = TIER_STYLE[r["tier"]]
            cost = TESTS[r["slot_test"]][2] if r["slot_test"] else 0
            why = reason(r, theta)
            html.append(
                f'<tr><td style="color:{STONE}">{i}</td>'
                f'<td>{r["patient_id"]}</td><td>{chip(r["tier"], fg, bg)}</td>'
                f'<td>{stage_track(int(r["stage"]), r["action"])}</td>'
                f'<td><b>{r["p"]:.2f}</b> <span style="color:{STONE}">± '
                f'{r["u"]:.2f}</span></td>'
                f'<td>{dot(ACTION_DOT[a])}{a.value}</td>'
                f'<td style="color:{STONE};line-height:1.4">{why}</td>'
                f'<td style="text-align:right">{"₹{:,}".format(cost) if cost else "—"}'
                f'</td></tr>')
        html.append("</tbody></table>")
        st.markdown("".join(html), unsafe_allow_html=True)

        v = top.reset_index(drop=True)
        inv = None
        for i in range(len(v)):
            for j in range(i + 1, len(v)):
                if v.loc[i, "p"] < v.loc[j, "p"] - 0.10:
                    inv = (v.loc[i], v.loc[j]); break
            if inv:
                break
        if inv:
            lo_, hi_ = inv
            st.markdown(
                f'<div class="note" style="margin-top:16px"><b>Risk inversion, on '
                f'purpose.</b> {lo_["patient_id"]} (risk {lo_["p"]:.2f}) outranks '
                f'{hi_["patient_id"]} (risk {hi_["p"]:.2f}) — the result is more '
                f'likely to land on the far side of θ for the first patient, so '
                f'the scan buys a decision rather than a confirmation.</div>',
                unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── policy + equity ──────────────────────────────────────────────
        c1, c2 = st.columns([1.7, 1])
        with c1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Same PET capacity, two ways to fill it")
            st.pyplot(fig_policy(s), use_container_width=True)
            if s["naive_pet"]:
                pctw = 100 * s["naive_pet_wasted"] / max(s["naive_pet"], 1)
                st.markdown(
                    f'<div class="note"><b>{s["naive_pet_wasted"]} of '
                    f'{s["naive_pet"]} ({pctw:.0f}%) risk-ranked PET slots go to '
                    f'patients where the scan cannot change the action.</b> '
                    f'Flip-filtering sends none.</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Equity guard")
            lo = plan[plan["low_education"]]; hi = plan[~plan["low_education"]]
            lr = float(lo["granted"].mean()) if len(lo) else 0.0
            hr = float(hi["granted"].mean()) if len(hi) else 0.0
            st.markdown(
                f'<div style="display:flex;gap:30px"><div><div style="color:{STONE};'
                f'font-size:15px">Low education</div><div style="font-size:30px">'
                f'{lr:.0%}</div></div><div><div style="color:{STONE};font-size:15px">'
                f'High education</div><div style="font-size:30px">{hr:.0%}</div>'
                f'</div></div>', unsafe_allow_html=True)
            if hr > 0 and lr < hr * 0.8:
                st.markdown('<div class="band" style="margin-top:14px">Queue may '
                            'under-serve low-education patients. MMSE is '
                            'education-biased and dementia prevalence is higher in '
                            'this group (10.29% vs 1.54%, IIPS national study).'
                            '</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="note" style="margin-top:14px">Selection '
                            'rates are within 20% of each other. MMSE remains '
                            'education-biased, so this runs on every allocation.'
                            '</div>', unsafe_allow_html=True)
            st.caption("Surfaced, never auto-corrected.")
            st.markdown('</div>', unsafe_allow_html=True)

    # ─────────────────────── architecture tab ────────────────────────────
    with tab_arch:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Reference architecture")
        st.pyplot(fig_architecture(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        a, b = st.columns(2)
        with a:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Cohort funnel")
            st.pyplot(fig_funnel(plan), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with b:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### Priority tiers")
            st.pyplot(fig_tiers(plan), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### Data provenance and de-identification")
        prov = pd.DataFrame([
            ["Cognitive (MMSE, CDR)", "OASIS-2 public cohort",
             "Real" if use_real else "Simulated"],
            ["MRI volumetrics (nWBV, eTIV, ASF)", "OASIS-2 public cohort",
             "Real" if use_real else "Simulated"],
            ["Co-morbidities, APOE4", "Simulated, seed 42", "Simulated"],
            ["Blood p-tau217", "Simulated from published assay performance",
             "Simulated"],
        ], columns=["Modality", "Source", "Status"])
        st.table(prov)
        st.markdown(
            '<div class="note">Patient identifiers are synthetic surrogates. No '
            'identifiable information is present anywhere in this prototype. '
            'The system accepts <b>MoCA</b> in place of MMSE through a published '
            'score mapping, so sites using either instrument can be onboarded '
            'without re-training.<br><br>'
            'Sensitivity, specificity and cost figures are published-literature '
            'estimates, not measured on this cohort. The training-free risk model '
            'uses a cognitive prior calibrated to a memory-clinic prevalence shape '
            '— the ordering is real, the calibration is an assumption.</div>',
            unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── export ───────────────────────────────────────────────────────────
    if export:
        FIG_DIR.mkdir(exist_ok=True)
        sel = plan[plan["patient_id"] == pick].iloc[0]
        expl = get_explainer(n, SEED, use_real)
        figs = {
            "funnel.png": fig_funnel(plan, True),
            "decision_flip.png": fig_flip(sel, theta, True),
            "coverage_grid.png": fig_coverage(plan, standalone=True),
            "policy_comparison.png": fig_policy(s, True),
            "architecture.png": fig_architecture(),
            "tiers.png": fig_tiers(plan, True),
            "factors.png": fig_factors(E.contributing_factors(sel, expl), True),
        }
        for name, f in figs.items():
            f.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight",
                      facecolor=WHITE)
            plt.close(f)
        st.sidebar.success(f"Wrote {len(figs)} figures to {FIG_DIR}")


def reason(r, theta):
    a = r["action"]
    if r["deferred"]:
        return "A test would decide this, but no slot is left this month."
    if a is Action.ESCALATE_NOW_PET_ELIGIBILITY:
        return ("Testing cannot move the triage call, but PET is required to "
                "confirm therapy eligibility.")
    if a is Action.ESCALATE_NOW:
        return (f"Every remaining test lands above θ={theta:.2f}. More testing "
                f"delays care without changing it.")
    if a is Action.RELEASE_12MO:
        return f"Every remaining test lands below θ={theta:.2f}."
    if a is Action.CLINICIAN_REVIEW:
        return ("Cognitive data only and the band still crosses θ. Too thin to "
                "release automatically.")
    k = r["slot_test"]
    e = r["evals"][k]
    return (f"Cheapest remaining test that crosses θ "
            f"({e['p_if_negative']:.2f} vs {e['p_if_positive']:.2f}), "
            f"{e['days']} days.")


if __name__ == "__main__":
    main()
