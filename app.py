from __future__ import annotations

import json
import html
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_workbook
from src.decision_intelligence_pipeline import DecisionIntelligencePipeline
from src.ollama_llm_engine import OllamaLLMEngine
from src.grounded_ai_engine import VARIAGroundedAIEngine

# ============================================================
# VARIA — FINAL CFO PRODUCT UI
# ============================================================

st.set_page_config(
    page_title="VARIA | FP&A Decision Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Design tokens ----------
INK = "#151B2F"
MUTED = "#8A93A6"
CANVAS = "#F5F3F4"
CARD = "#FFFFFF"
RAIL = "#111827"
PINK = "#EF4F86"
PINK_DARK = "#D93E73"
PINK_SOFT = "#FFE3ED"
BLUE = "#DDEAFF"
LAVENDER = "#8A7BDA"
LAVENDER_SOFT = "#ECE8FF"
GREEN = "#70B98A"
GREEN_SOFT = "#E6F5EB"
AMBER = "#F3BC4F"
AMBER_SOFT = "#FFF1CE"
RED = "#DE5D6B"
RED_SOFT = "#FFE2E7"
LINE = "#ECE9EE"
NAVY_SOFT = "#EEF1F6"

st.markdown(
    f"""
<style>
:root {{
  --ink:{INK}; --muted:{MUTED}; --canvas:{CANVAS}; --card:{CARD}; --rail:{RAIL};
  --pink:{PINK}; --pink-soft:{PINK_SOFT}; --blue:{BLUE}; --lav:{LAVENDER};
  --green:{GREEN}; --green-soft:{GREEN_SOFT}; --amber:{AMBER}; --amber-soft:{AMBER_SOFT};
  --red:{RED}; --red-soft:{RED_SOFT}; --line:{LINE};
}}
html,body,[class*="css"] {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
.stApp {{
  background:
    radial-gradient(circle at 98% 2%, rgba(207,198,255,.34), transparent 24%),
    radial-gradient(circle at 2% 0%, rgba(202,224,255,.24), transparent 18%),
    var(--canvas);
  color:var(--ink);
}}
[data-testid="stHeader"] {{ background:transparent !important; height:0 !important; }}
section[data-testid="stSidebar"] {{ display:none !important; }}
#stDecoration, [data-testid="stAppDeployButton"], [data-testid="stToolbarActions"], button[title*="Deploy"] {{ display:none !important; visibility:hidden !important; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{ display:none !important; visibility:hidden !important; }}
.block-container {{ max-width: 1580px; padding: 10px 26px 62px; }}
.stAppDeployButton {{ display:none !important; }}


/* Top product header */
.topline {{ display:flex; justify-content:space-between; align-items:center; margin:3px 0 10px; color:#8D95A6; font-size:10px; font-weight:900; letter-spacing:1.4px; text-transform:uppercase; }}
.brand-row {{ display:flex; justify-content:space-between; align-items:center; gap:20px; margin-bottom:12px; }}
.brand {{ display:flex; align-items:center; gap:12px; }}
.brand-mark {{ width:44px; height:44px; border-radius:14px; background:{RAIL}; color:#fff; display:flex; align-items:center; justify-content:center; font-size:18px; font-weight:950; box-shadow:0 8px 24px rgba(17,24,39,.12); }}
.brand-name {{ font-size:19px; font-weight:950; letter-spacing:.3px; }}
.brand-sub {{ font-size:10px; color:#9AA2B1; font-weight:850; letter-spacing:1.2px; text-transform:uppercase; margin-top:2px; }}
.header-right {{ display:flex; align-items:center; gap:8px; }}
.header-pill {{ background:#fff; border:1px solid var(--line); border-radius:999px; padding:9px 13px; color:#757E90; font-size:11px; box-shadow:0 6px 18px rgba(20,25,45,.045); }}

/* Navigation */
.nav-wrap {{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:8px; box-shadow:0 12px 32px rgba(20,25,45,.055); margin-bottom:12px; }}
.nav-caption {{ color:#A0A7B5; font-size:8px; font-weight:950; letter-spacing:1.25px; text-transform:uppercase; padding:2px 10px 8px; }}
.nav-wrap .stButton > button {{ min-height:38px !important; padding:6px 5px !important; border-radius:13px !important; border:1px solid #E9E6EA !important; background:#fff !important; color:#626C7D !important; font-size:8.5px !important; font-weight:900 !important; box-shadow:none !important; white-space:nowrap !important; overflow:hidden !important; text-overflow:ellipsis !important; }}
.nav-wrap .stButton > button:hover {{ background:#F7F4F6 !important; color:{INK} !important; border-color:#DDD8DF !important; }}
.nav-wrap .stButton > button[kind="primary"] {{ background:{PINK} !important; color:#fff !important; border-color:{PINK} !important; }}

/* Control strip */
.control-shell {{ background:#fff; border:1px solid var(--line); border-radius:20px; padding:11px 12px 8px; box-shadow:0 8px 24px rgba(20,25,45,.045); margin-bottom:16px; }}
.control-label {{ font-size:8px; color:#8E96A6; font-weight:950; letter-spacing:1.2px; text-transform:uppercase; margin:1px 0 5px; }}
.control-shell .stSelectbox > div > div, .control-shell .stTextInput > div > div {{ border-radius:12px !important; border:1px solid #ECE9EE !important; background:#FBFAFB !important; min-height:38px !important; }}
.control-shell .stFileUploader {{ padding:0 !important; }}
.control-shell .stFileUploader > div {{ border:1px dashed #DDD8E0 !important; border-radius:12px !important; background:#FBFAFB !important; }}

/* Head */
.page-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px; margin:5px 2px 12px; }}
.eyebrow {{ color:#81899B; font-size:10px; font-weight:950; letter-spacing:1.65px; text-transform:uppercase; }}
.page-title {{ color:{INK}; font-size:54px; font-weight:950; line-height:.98; letter-spacing:-3px; margin-top:6px; }}
.page-sub {{ color:#7D8698; font-size:14px; line-height:1.45; margin-top:9px; max-width:900px; }}
.meta-row {{ margin-top:11px; display:flex; flex-wrap:wrap; gap:6px; }}
.badge {{ display:inline-flex; align-items:center; border:1px solid var(--line); background:#fff; border-radius:999px; padding:6px 10px; color:#687284; font-size:9px; font-weight:900; }}

/* Cards */
.card {{ background:#fff; border:1px solid rgba(20,25,45,.035); border-radius:22px; padding:19px; box-shadow:0 8px 24px rgba(20,25,45,.045); }}
.card-tight {{ padding:14px 16px; }}
.pink-card {{ background:linear-gradient(135deg,{PINK} 0%, #F35C8F 100%); color:#fff; }}
.blue-card {{ background:{BLUE}; }}
.soft-card {{ background:#FBFAFB; }}
.green-card {{ background:{GREEN_SOFT}; }}
.amber-card {{ background:{AMBER_SOFT}; }}
.red-card {{ background:{RED_SOFT}; }}
.kicker {{ color:#929AAA; font-size:9px; font-weight:950; letter-spacing:1.2px; text-transform:uppercase; }}
.card-title {{ color:{INK}; font-size:18px; font-weight:950; letter-spacing:-.4px; margin-top:4px; }}
.card-value {{ color:{INK}; font-size:34px; font-weight:950; letter-spacing:-1.8px; line-height:1.03; margin-top:8px; }}
.card-value.sm {{ font-size:24px; }}
.sub {{ color:#858E9F; font-size:11px; line-height:1.52; margin-top:6px; }}
.pink-card .kicker,.pink-card .sub,.pink-card .card-title {{ color:rgba(255,255,255,.88); }}
.pink-card .card-value {{ color:#fff; }}

/* Metric grid */
.metric-grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:10px; }}
.metric {{ background:#fff; border:1px solid var(--line); border-radius:18px; padding:14px 15px; min-height:102px; box-shadow:0 6px 18px rgba(20,25,45,.035); }}
.metric .label {{ color:#8C95A6; font-size:9px; font-weight:900; letter-spacing:.55px; }}
.metric .value {{ color:{INK}; font-size:25px; font-weight:950; letter-spacing:-1.1px; margin-top:8px; }}
.metric .note {{ color:#9AA2AF; font-size:9px; margin-top:4px; }}

/* Sections */
.section-head {{ display:flex; justify-content:space-between; align-items:flex-end; margin:22px 2px 8px; }}
.section-title {{ color:{INK}; font-size:18px; font-weight:950; letter-spacing:-.5px; }}
.section-note {{ color:#9098A8; font-size:10px; margin-top:3px; }}

/* Charts */
.chart-card {{ background:#fff; border:1px solid var(--line); border-radius:22px; padding:12px 15px 4px; box-shadow:0 7px 20px rgba(20,25,45,.035); }}
.chart-title {{ color:{INK}; font-size:15px; font-weight:950; }}
.chart-note {{ color:#929AAA; font-size:10px; margin-top:2px; margin-bottom:4px; }}

/* Status pills */
.status {{ display:inline-flex; align-items:center; border-radius:999px; padding:6px 10px; font-size:9px; font-weight:950; white-space:nowrap; }}
.status-red {{ background:{RED_SOFT}; color:#C84759; }}
.status-amber {{ background:{AMBER_SOFT}; color:#A66E00; }}
.status-green {{ background:{GREEN_SOFT}; color:#428760; }}
.status-pink {{ background:{PINK_SOFT}; color:{PINK_DARK}; }}
.status-blue {{ background:#E8EEFF; color:#5870B0; }}
.status-neutral {{ background:#EFF1F5; color:#687284; }}

/* Issue cards */
.issue {{ background:#fff; border:1px solid var(--line); border-radius:19px; padding:15px 16px; margin-top:9px; box-shadow:0 6px 18px rgba(20,25,45,.035); }}
.issue-top {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
.issue-kicker {{ color:#8E97A7; font-size:9px; font-weight:950; letter-spacing:1.1px; text-transform:uppercase; }}
.issue-name {{ color:{INK}; font-size:15px; font-weight:950; margin-top:4px; }}
.issue-metrics {{ display:grid; grid-template-columns:repeat(3,1fr); gap:10px; margin-top:13px; }}
.issue-metrics .lbl {{ color:#98A0AE; font-size:8px; font-weight:900; text-transform:uppercase; letter-spacing:1px; }}
.issue-metrics .val {{ color:{INK}; font-size:18px; font-weight:950; margin-top:4px; }}
.issue-action {{ color:#717A8C; font-size:10px; line-height:1.5; margin-top:12px; }}

/* AI */
.ai-grid {{ display:grid; grid-template-columns:repeat(6,1fr); gap:9px; }}
.ai-check {{ background:#fff; border:1px solid var(--line); border-radius:15px; padding:11px 12px; }}
.ai-check .v {{ color:{INK}; font-size:20px; font-weight:950; margin-top:3px; }}
.ai-check.pass .v {{ color:{GREEN}; }}
.ai-check.fail .v {{ color:{RED}; }}
.evidence-row {{ background:#fff; border:1px solid var(--line); border-radius:16px; padding:13px 14px; margin-top:8px; }}
.evidence-id {{ color:#8D96A6; font-size:9px; font-weight:950; letter-spacing:.6px; }}
.evidence-text {{ color:#5F687B; font-size:11px; line-height:1.55; margin-top:5px; }}

/* small insights */
.insight-box {{ background:#F8F7FA; border:1px solid var(--line); border-radius:17px; padding:13px 14px; color:#626C7E; font-size:11px; line-height:1.6; }}

/* hide plotly toolbar */
.js-plotly-plot .modebar {{ display:none !important; }}
.js-plotly-plot .plotly .axis text {{ font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important; }}
.stButton > button {{ transition: all .12s ease; }}

/* dataframes */
[data-testid="stDataFrame"] {{ border:1px solid var(--line); border-radius:16px; overflow:hidden; }}

/* responsive */
@media(max-width:1100px) {{ .metric-grid{{grid-template-columns:repeat(3,1fr)}} .ai-grid{{grid-template-columns:repeat(3,1fr)}} .page-title{{font-size:44px}} }}
@media(max-width:700px) {{ .block-container{{padding:10px 12px 35px}} .metric-grid{{grid-template-columns:repeat(2,1fr)}} .ai-grid{{grid-template-columns:repeat(2,1fr)}} .page-head{{flex-direction:column;align-items:flex-start}} .page-title{{font-size:38px}} }}

.static-source {{
    background: rgba(255,255,255,.78);
    border: 1px solid rgba(21,27,47,.08);
    border-radius: 14px;
    padding: 11px 13px;
    color: #5E687A;
    font-size: 12px;
    min-height: 40px;
    display: flex;
    align-items: center;
}}
</style>
""",
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

def num(x, default=0.0):
    try:
        y = float(x)
        return default if pd.isna(y) else y
    except Exception:
        return default


def money(x):
    x = num(x)
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e7:
        return f"{sign}₹{a/1e7:.2f} Cr"
    if a >= 1e5:
        return f"{sign}₹{a/1e5:.2f} L"
    if a >= 1e3:
        return f"{sign}₹{a/1e3:.1f} K"
    return f"{sign}₹{a:,.0f}"


def pct(x):
    return f"{num(x):.2f}%"


def money_compact(x):
    return money(x).replace("₹", "₹")


def safe_dt(df, col):
    x = df.copy()
    if col in x.columns:
        x[col] = pd.to_datetime(x[col], errors="coerce")
    return x


def esc(x):
    return html.escape(str(x))


def status_html(label):
    x = str(label).upper()
    cls = {
        "RED": "status-red", "CRITICAL": "status-red", "HIGH PRIORITY": "status-red",
        "AMBER": "status-amber", "REVIEW": "status-amber",
        "GREEN": "status-green", "MONITOR": "status-green", "PASS": "status-green",
        "RECOMMENDED": "status-pink", "FAVORABLE": "status-green", "UNFAVORABLE": "status-red",
    }.get(x, "status-neutral")
    return f"<span class='status {cls}'>{esc(label)}</span>"


def metric_grid(items):
    blocks = []
    for label, value, note in items:
        blocks.append(f"<div class='metric'><div class='label'>{esc(label)}</div><div class='value'>{esc(value)}</div><div class='note'>{esc(note)}</div></div>")
    st.markdown(f"<div class='metric-grid'>{''.join(blocks)}</div>", unsafe_allow_html=True)


def card(kicker, value, sub, cls="card"):
    return f"<div class='card {cls}'><div class='kicker'>{esc(kicker)}</div><div class='card-value'>{esc(value)}</div><div class='sub'>{esc(sub)}</div></div>"


def section(title, note=""):
    st.markdown(f"<div class='section-head'><div><div class='section-title'>{esc(title)}</div><div class='section-note'>{esc(note)}</div></div></div>", unsafe_allow_html=True)


def chart(title, note, fig, height=330, key=None):
    st.markdown(f"<div class='chart-card'><div class='chart-title'>{esc(title)}</div><div class='chart-note'>{esc(note)}</div></div>", unsafe_allow_html=True)
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=18, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#737D90", size=10),
        legend=dict(orientation="h", y=1.05, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=9)),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#fff")),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=LINE)
    fig.update_yaxes(showgrid=True, gridcolor="#F0EEF2", zeroline=False)
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False})


def download(df, label, filename):
    if isinstance(df, pd.DataFrame):
        st.download_button(label, df.to_csv(index=False).encode("utf-8"), filename, "text/csv", width="stretch")


def normalize_columns(df):
    x = df.copy()
    x.columns = [str(c).strip().lower().replace(" ", "_").replace("-", "_") for c in x.columns]
    aliases = {
        "date": ["date", "transaction_date", "transactiondate", "period", "month"],
        "transaction_id": ["transaction_id", "transactionid", "txn_id", "id"],
        "business_unit": ["business_unit", "businessunit", "bu", "business"],
        "department": ["department", "dept", "function"],
        "category": ["category", "expense_category", "expense_type", "cost_category"],
        "region": ["region", "geography", "zone"],
        "cost_centre": ["cost_centre", "cost_center", "costcentre", "costcenter"],
        "vendor": ["vendor", "supplier"],
        "actual_amount": ["actual_amount", "actual", "actuals", "expense", "amount", "spend"],
        "budget_amount": ["budget_amount", "budget", "plan", "planned", "budgeted_amount"],
        "status": ["status", "transaction_status"],
    }
    rename = {}
    for target, choices in aliases.items():
        if target in x.columns:
            continue
        for c in choices:
            if c in x.columns:
                rename[c] = target
                break
    return x.rename(columns=rename)


@st.cache_data(show_spinner=False)
def default_data():
    d = load_workbook()
    return d["Budget"].copy(), d["Actuals_Raw"].copy()


@st.cache_data(show_spinner=False)
def compute(budget, actuals):
    return DecisionIntelligencePipeline(budget=budget, actuals=actuals).run(enable_local_llm=False)


def derive_risk_lookup(risk_df):
    if not isinstance(risk_df, pd.DataFrame) or risk_df.empty:
        return pd.DataFrame()
    keys = [c for c in ["Business_Unit","Department","Category"] if c in risk_df.columns]
    vals = [c for c in ["Risk_Score","Risk_Band","Risk_Rank"] if c in risk_df.columns]
    if not keys or not vals:
        return pd.DataFrame()
    return risk_df[keys+vals].drop_duplicates(keys)


def merge_risk_into_queue(q, risk_df):
    """Attach risk-engine fields to CFO queue without changing queue facts."""
    q = q.copy()
    if q.empty:
        return q
    rl = derive_risk_lookup(risk_df)
    if rl.empty:
        return q
    keys = [c for c in ["Business_Unit", "Department", "Category"] if c in q.columns and c in rl.columns]
    if len(keys) != 3:
        return q
    # Remove stale risk columns so the merge cannot create confusing *_risk duplicates.
    q = q.drop(columns=[c for c in ["Risk_Score", "Risk_Band", "Risk_Rank"] if c in q.columns], errors="ignore")
    return q.merge(rl, on=keys, how="left")


def attach_verified_current_variance(q, variance_scope):
    """Attach authoritative current variance without a pandas merge collision."""
    q = q.copy()
    if q.empty or not isinstance(variance_scope, pd.DataFrame) or variance_scope.empty:
        return q
    keys = ["Business_Unit", "Department", "Category"]
    if not all(c in q.columns and c in variance_scope.columns for c in keys):
        return q
    if "Variance" not in variance_scope.columns:
        return q

    def norm(x):
        if pd.isna(x):
            return "__MISSING__"
        return str(x).strip().casefold()

    lookup = {}
    for _, row in variance_scope.iterrows():
        key = tuple(norm(row[c]) for c in keys)
        lookup[key] = lookup.get(key, 0.0) + num(row.get("Variance", 0))

    q["Current_Variance"] = [
        lookup.get(tuple(norm(row.get(c)) for c in keys), 0.0)
        for _, row in q.iterrows()
    ]
    q["Current_Variance"] = pd.to_numeric(q["Current_Variance"], errors="coerce").fillna(0.0)
    return q


def scenario_selected(sc, summary):
    if not isinstance(sc, pd.DataFrame) or sc.empty:
        return None
    rec = summary.get("Recommended_Scenario", summary.get("recommended_scenario")) if isinstance(summary, dict) else None
    if rec is None and "Scenario_Name" in sc.columns:
        rec = sc.iloc[-1]["Scenario_Name"]
    if rec is not None and "Scenario_Name" in sc.columns:
        hit = sc[sc["Scenario_Name"].astype(str) == str(rec)]
        if len(hit):
            return hit.iloc[0]
    return sc.iloc[-1]


def fmt_scenario_summary(sc, summary):
    row = scenario_selected(sc, summary)
    if row is None:
        return "Base Scenario", 0, 0, 0
    name = str(row.get("Scenario_Name", "Scenario"))
    basev = num(row.get("Base_Projected_Variance", 0))
    scenv = num(row.get("Scenario_Projected_Variance", 0))
    improvement = num(row.get("Variance_Improvement", row.get("Forecast_Savings", basev-scenv)))
    return name, basev, scenv, improvement



# ============================================================
# AUTHORITATIVE CURRENT-VARIANCE DISPLAY LOOKUP
# ============================================================
def verified_current_variance(row, variance_df):
    if not isinstance(variance_df, pd.DataFrame) or variance_df.empty:
        return 0.0
    keys = ["Business_Unit", "Department", "Category"]
    if not all(c in variance_df.columns for c in keys + ["Variance"]):
        return 0.0

    def norm(x):
        if pd.isna(x):
            return "__MISSING__"
        return str(x).strip().casefold()

    mask = pd.Series(True, index=variance_df.index)
    for col in keys:
        if col not in row.index:
            return 0.0
        mask = mask & variance_df[col].map(norm).eq(norm(row[col]))
    vals = pd.to_numeric(variance_df.loc[mask, "Variance"], errors="coerce").fillna(0.0)
    return float(vals.sum()) if len(vals) else 0.0

# ============================================================
# NAV / STATE
# ============================================================
PAGES = [
    ("Executive","⌂"),("KPI Cockpit","◉"),("Analytics","▦"),("Statistics","∿"),
    ("Forecast & Risk","⌁"),("Scenarios","◇"),("CFO Queue","▣"),("AI Insights","✦"),("Recommendations","✓"),
    ("Root Cause","⌖"),("Data & Audit","▤"),
]
if "page" not in st.session_state:
    st.session_state.page = "Executive"
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None

st.markdown("<div class='topline'><span>FP&A • Forecasting • Risk • AI</span><span>● Local analytics</span></div>", unsafe_allow_html=True)
st.markdown("<div class='brand-row'><div class='brand'><div class='brand-mark'>◆</div><div><div class='brand-name'>VARIA</div><div class='brand-sub'>CFO Decision Intelligence</div></div></div><div class='header-right'><div class='header-pill'>FY23-24 → FY26-27</div><div class='header-pill'>Local model</div></div></div>", unsafe_allow_html=True)

st.markdown("<div class='nav-wrap'><div class='nav-caption'>Workspace</div>", unsafe_allow_html=True)
nav_labels = {
    "Executive":"Executive", "KPI Cockpit":"KPI", "Analytics":"Analytics", "Statistics":"Stats",
    "Forecast & Risk":"Forecast", "Scenarios":"Scenarios", "CFO Queue":"CFO Queue", "AI Insights":"AI Insights",
    "Recommendations":"Recs", "Root Cause":"Root Cause", "Data & Audit":"Data", 
}
nav_cols = st.columns(len(PAGES)+1)
for col,(name,icon) in zip(nav_cols[:-1], PAGES):
    with col:
        label = f"{icon}  {nav_labels.get(name, name)}"
        if st.button(label, width="stretch", key=f"nav_{name}", type="primary" if st.session_state.page==name else "secondary"):
            st.session_state.page = name
            st.rerun()
with nav_cols[-1]:
    if st.button("💬  Ask", width="stretch", key="nav_chat", type="primary" if st.session_state.chat_open else "secondary"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# DATA + FILTERS — FINAL SUBMISSION MODE
# ============================================================
# VARIA is intentionally locked to the verified project dataset for
# submission. External workbook/CSV upload is not exposed in the UI.
# Universal ingestion/mapping remains available in the backend modules.

source_label = "Current verified dataset"

try:
    budget, actuals = default_data()
    outputs = compute(budget, actuals)
except Exception:
    st.error("VARIA could not build the financial model from the verified project dataset.")
    st.stop()

# Filter options are derived only from the verified dataset.
variance_preview = outputs.get("variance", pd.DataFrame()).copy()
bus = sorted(variance_preview.get("Business_Unit", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if isinstance(variance_preview, pd.DataFrame) else []
deps = sorted(variance_preview.get("Department", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if isinstance(variance_preview, pd.DataFrame) else []
cats = sorted(variance_preview.get("Category", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if isinstance(variance_preview, pd.DataFrame) else []
fys = sorted(variance_preview.get("Fiscal_Year", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()) if isinstance(variance_preview, pd.DataFrame) else []

st.markdown("<div class='control-shell'>", unsafe_allow_html=True)
control_cols = st.columns([1.0,1.0,1.0,1.0,1.8,1.3])
with control_cols[0]:
    st.markdown("<div class='control-label'>Business unit</div>", unsafe_allow_html=True)
    selected_bu = st.selectbox("BU", ["All"] + bus, label_visibility="collapsed", key="bu")
with control_cols[1]:
    st.markdown("<div class='control-label'>Department</div>", unsafe_allow_html=True)
    selected_dept = st.selectbox("Department", ["All"] + deps, label_visibility="collapsed", key="dept")
with control_cols[2]:
    st.markdown("<div class='control-label'>Category</div>", unsafe_allow_html=True)
    selected_cat = st.selectbox("Category", ["All"] + cats, label_visibility="collapsed", key="cat")
with control_cols[3]:
    st.markdown("<div class='control-label'>Fiscal year</div>", unsafe_allow_html=True)
    selected_fy = st.selectbox("Fiscal year", ["All"] + fys, label_visibility="collapsed", key="fy")
with control_cols[4]:
    st.markdown("<div class='control-label'>Global search</div>", unsafe_allow_html=True)
    search = st.text_input("Search", placeholder="Search issues, drivers, categories…", label_visibility="collapsed", key="global_search")
with control_cols[5]:
    st.markdown("<div class='control-label'>Data source</div>", unsafe_allow_html=True)
    st.markdown("<div class='static-source'>✓ Current verified dataset</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

variance = outputs.get("variance", pd.DataFrame()).copy()
materiality = outputs.get("materiality", pd.DataFrame()).copy()
forecast = outputs.get("forecast", pd.DataFrame()).copy()
warning = outputs.get("warning", pd.DataFrame()).copy()
cfo = merge_risk_into_queue(outputs.get("cfo_queue", pd.DataFrame()).copy(), outputs.get("risk", pd.DataFrame()).copy())
root_cause = outputs.get("root_cause", pd.DataFrame()).copy()
anomaly = outputs.get("anomaly", pd.DataFrame()).copy()
risk = outputs.get("risk", pd.DataFrame()).copy()
recommendations = outputs.get("recommendations", pd.DataFrame()).copy()
action_sizing = outputs.get("action_sizing", pd.DataFrame()).copy()
scenarios = outputs.get("scenario_comparison", pd.DataFrame()).copy()
scenario_summary = outputs.get("scenario_summary", {}) or {}
advanced_forecast = outputs.get("advanced_forecast", pd.DataFrame()).copy()
audit = outputs.get("audit", pd.DataFrame()).copy()
audit_summary = outputs.get("audit_summary", {}) or {}
evidence_pack = outputs.get("cfo_evidence_pack") or {}
cfo_report = outputs.get("cfo_report", "")
ai_base = outputs.get("ai_output", pd.DataFrame()).copy()
planning_summary = outputs.get("planning_summary", {}) or {}
planning_exceptions = outputs.get("planning_exceptions", pd.DataFrame()).copy()


def filt(df):
    x = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if x.empty:
        return x
    for col, val in [("Business_Unit", selected_bu), ("Department", selected_dept), ("Category", selected_cat), ("Fiscal_Year", selected_fy)]:
        if val != "All" and col in x.columns:
            x = x[x[col].astype(str) == str(val)]
    if search:
        mask = pd.Series(False, index=x.index)
        for col in x.columns:
            try:
                mask = mask | x[col].astype(str).str.contains(search, case=False, na=False, regex=False)
            except Exception:
                pass
        x = x[mask]
    return x

v = filt(variance)
m = filt(materiality)
f = filt(forecast)
w = filt(warning)
q = filt(cfo)
q = attach_verified_current_variance(q, v)
rsk = filt(risk)
r = filt(root_cause)

def filt(df):
    x = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if x.empty:
        return x
    for col, val in [("Business_Unit", selected_bu), ("Department", selected_dept), ("Category", selected_cat), ("Fiscal_Year", selected_fy)]:
        if val != "All" and col in x.columns:
            x = x[x[col].astype(str) == str(val)]
    if search:
        mask = pd.Series(False, index=x.index)
        for col in x.columns:
            try:
                mask = mask | x[col].astype(str).str.contains(search, case=False, na=False, regex=False)
            except Exception:
                pass
        x = x[mask]
    return x

v = filt(variance)
m = filt(materiality)
f = filt(forecast)
w = filt(warning)
q = filt(cfo)
# Current variance is authoritative from the Variance Engine at the selected scope.
q = attach_verified_current_variance(q, v)
rsk = filt(risk)
r = filt(root_cause)

# ============================================================
# ASK VARIA CHATBOT — persistent top panel
# ============================================================

def deterministic_answer(question):
    qn = question.lower().strip()
    b = num(v.get("Budget_Amount", pd.Series(dtype=float)).sum())
    a = num(v.get("Actual_Amount", pd.Series(dtype=float)).sum())
    var = num(v.get("Variance", pd.Series(dtype=float)).sum())
    unf = num(v.loc[v.get("Variance", pd.Series(dtype=float)) > 0, "Variance"].sum()) if "Variance" in v else 0
    fav = abs(num(v.loc[v.get("Variance", pd.Series(dtype=float)) < 0, "Variance"].sum())) if "Variance" in v else 0
    if any(k in qn for k in ["total variance","net variance","overall variance"]):
        return f"The selected scope has a net unfavorable variance of {money(var)} ({pct(var/b*100 if b else 0)} of budget)."
    if "budget" in qn and "actual" in qn:
        return f"Budget is {money(b)} versus matched actuals of {money(a)}, giving utilization of {pct(a/b*100 if b else 0)}."
    if "unfavorable" in qn:
        return f"Unfavorable exposure is {money(unf)}, partially offset by {money(fav)} of favorable variance, leaving {money(var)} net unfavorable."
    if "material" in qn:
        nmat = int(m["Is_Material"].sum()) if "Is_Material" in m else 0
        return f"VARIA flags {nmat:,} material planning combinations in the selected scope."
    if "exception" in qn:
        return f"VARIA found {int(planning_summary.get('Exception Transactions',0)):,} planning exceptions with {money(planning_summary.get('Exception Amount',0))} of exception exposure."
    if "red" in qn or "early warning" in qn:
        redc = int((w["Early_Warning"].astype(str).str.upper()=="RED").sum()) if "Early_Warning" in w else 0
        return f"There are {redc:,} RED early-warning records in the selected scope."
    if "risk" in qn and "biggest" in qn or "highest risk" in qn:
        if len(rsk) and "Risk_Score" in rsk:
            rr = rsk.sort_values("Risk_Score", ascending=False).iloc[0]
            return f"The highest risk item is {rr.get('Business_Unit','—')} / {rr.get('Department','—')} / {rr.get('Category','—')} with a risk score of {num(rr.get('Risk_Score',0)):.0f} ({rr.get('Risk_Band','—')})."
    # Common dimension-level financial questions.
    dimension_defs = [
        ("business unit", "Business_Unit"),
        ("department", "Department"),
        ("category", "Category"),
    ]
    for label, col in dimension_defs:
        if col not in v.columns:
            continue
        values = v[col].dropna().astype(str).unique().tolist()
        for value in values:
            if value.lower() in qn:
                mask = v[col].astype(str).str.lower() == value.lower()
                sub = v.loc[mask]
                if sub.empty:
                    continue
                if "budget" in qn:
                    amount = num(sub["Budget_Amount"].sum()) if "Budget_Amount" in sub else 0
                    return f"The budget for {label} {value} is {money(amount)}."
                if "actual" in qn or "spend" in qn:
                    amount = num(sub["Actual_Amount"].sum()) if "Actual_Amount" in sub else 0
                    return f"Matched actual spend for {label} {value} is {money(amount)}."
                if "variance" in qn:
                    amount = num(sub["Variance"].sum()) if "Variance" in sub else 0
                    return f"The variance for {label} {value} is {money(amount)}."

    if "driver" in qn and ("top" in qn or "biggest" in qn or "driving" in qn):
        if len(r):
            imp = "Unfavorable_Variance" if "Unfavorable_Variance" in r.columns else None
            rr = r.sort_values(imp, ascending=False).iloc[0] if imp else r.iloc[0]
            return f"The leading recurring driver is {rr.get('Department','—')} / {rr.get('Category','—')} with {money(rr.get(imp,0)) if imp else 'material unfavorable exposure'}."
    if "scenario" in qn and ("best" in qn or "recommended" in qn):
        name, basev, scenv, improvement = fmt_scenario_summary(scenarios, scenario_summary)
        return f"The currently recommended scenario is {name}, with modeled improvement of {money(improvement)} versus a base projected variance of {money(basev)}."
    return None


if st.session_state.chat_open:
    st.markdown("<div class='card' style='margin-bottom:14px'><div class='kicker'>VARIA COPILOT</div><div class='card-title'>Ask your financial model</div><div class='sub'>Core KPI questions are answered deterministically from VARIA. Broader questions are grounded in the verified evidence pack and sent to local Qwen only when needed.</div>", unsafe_allow_html=True)
    quick_cols = st.columns(5)
    quick = ["What is the total variance?","What is driving the overspend?","Which area is highest risk?","How many planning exceptions?","Which scenario is recommended?"]
    for i,qq in enumerate(quick):
        with quick_cols[i]:
            if st.button(qq, key=f"quick_{i}", width="stretch"):
                st.session_state.chat_history.append(("user", qq))
                ans = deterministic_answer(qq)
                if ans is None:
                    ans = "I need broader evidence-grounded analysis for that question."
                st.session_state.chat_history.append(("assistant", ans))
                st.rerun()
    for role,msg in st.session_state.chat_history[-8:]:
        bubble = "background:#F3F1F5" if role=="user" else f"background:#EAF0FF"
        align = "margin-left:18%" if role=="user" else "margin-right:18%"
        st.markdown(f"<div style='padding:10px 12px;border-radius:15px;margin-top:7px;{align};{bubble};color:#5E687A;font-size:11px;line-height:1.55'><b>{'You' if role=='user' else 'VARIA'}</b><br>{esc(msg)}</div>", unsafe_allow_html=True)
    c1,c2=st.columns([5.5,1])
    with c1:
        question = st.text_input("Question", placeholder="e.g. What is driving the overspend? Which department is the biggest risk?", label_visibility="collapsed", key="chat_q")
    with c2:
        ask = st.button("Ask", type="primary", width="stretch", key="chat_ask")
    if ask and question.strip():
        qtext = question.strip()
        ans = deterministic_answer(qtext)
        if ans is None:
            try:
                compact = {
                    "executive_position": evidence_pack.get("executive_position",{}),
                    "top_drivers": evidence_pack.get("top_drivers",[]),
                    "top_cfo_issues": evidence_pack.get("top_cfo_issues",[]),
                    "planning_exceptions": evidence_pack.get("planning_exceptions",{}),
                    "scenario_evidence": evidence_pack.get("scenario_evidence",{}),
                }
                client = OllamaLLMEngine(model="qwen2.5:7b", timeout=180)
                prompt_text = (
                    "Answer the user's question using ONLY this verified VARIA evidence. "
                    "Do not invent numbers, causes, trends, or financial meaning. "
                    "If evidence is insufficient, say so. Use a concise management-style answer.\n\n"
                    "QUESTION:\n" + qtext + "\n\nVERIFIED EVIDENCE:\n" +
                    json.dumps(compact, default=str)
                )
                # OllamaLLMEngine expects context_payload, not prompt=.
                resp = client.generate(context_payload=prompt_text, temperature=0.0)
                ans = resp.get("response","") if isinstance(resp,dict) else str(resp)
            except Exception as exc:
                ans = f"I could not reach local Qwen. VARIA deterministic analytics remain available. ({exc})"
        st.session_state.chat_history.append(("user", qtext)); st.session_state.chat_history.append(("assistant", ans)); st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE HEAD
# ============================================================
PAGE_SUB = {
    "Executive":"CFO command centre — what happened, what matters, and where to act",
    "KPI Cockpit":"Core financial health indicators for the selected scope",
    "Analytics":"Drivers, concentration, heatmaps, distributions and financial patterns",
    "Statistics":"Dispersion, volatility, outliers and statistical diagnostics",
    "Forecast & Risk":"Forward exposure, forecast trajectory and risk prioritization",
    "Scenarios":"Model intervention outcomes before committing resources",
    "CFO Queue":"Prioritized issues with management action context",
    "AI Insights":"Evidence-grounded commentary and model governance",
    "Recommendations":"Management actions ranked by financial impact and priority",
    "Root Cause":"Recurring drivers, exposure concentration and evidence-supported interpretation",
    "Data & Audit":"Data lineage, reconciliation, exceptions, governance and exports",
}
page = st.session_state.page
st.markdown(
    f"<div class='page-head'><div><div class='eyebrow'>FP&A • Forecasting • Risk • AI</div><div class='page-title'>{esc(page)}</div><div class='page-sub'>{esc(PAGE_SUB[page])}</div><div class='meta-row'><span class='badge'>Source: {esc(source_label)}</span><span class='badge'>Scope: {esc(selected_bu)} · {esc(selected_dept)} · {esc(selected_cat)} · {esc(selected_fy)}</span></div></div><div class='header-right'><div class='header-pill'>⌕ Global search</div><div class='header-pill'>{esc(source_label)}</div></div></div>",
    unsafe_allow_html=True,
)

# Common monthly series
vx = safe_dt(v, "Month") if "Month" in v.columns else v.copy()
if len(vx) and "Budget_Amount" in vx.columns and "Actual_Amount" in vx.columns and "Variance" in vx.columns:
    monthly = vx.groupby("Month", as_index=False).agg(Budget=("Budget_Amount","sum"),Actual=("Actual_Amount","sum"),Variance=("Variance","sum")).sort_values("Month")
else:
    monthly = pd.DataFrame()

# ============================================================
# PRESENTATION HELPERS
# ============================================================
def recommendation_rows(df, limit=8):
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    x = df.copy()
    sort_col = next((c for c in ["Priority_Score","Priority","Impact_Score","Projected_Variance","Current_Variance"] if c in x.columns), None)
    if sort_col:
        try: x = x.sort_values(sort_col, ascending=False)
        except Exception: pass
    return list(x.head(limit).iterrows())


def rec_text(row):
    for c in ["Recommendation","Management_Recommendation","Recommended_Action","Management_Action","Action","Recommendation_Text"]:
        if c in row.index and pd.notna(row[c]) and str(row[c]).strip():
            return str(row[c])
    return "Investigate the driver, validate business need, and assess corrective action or budget reallocation."


def priority_text(row):
    for c in ["Priority","Decision_Priority","Risk_Band","Severity"]:
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return "REVIEW"


def safe_col(df, candidates, default=None):
    for c in candidates:
        if isinstance(df, pd.DataFrame) and c in df.columns:
            return c
    return default


def management_insights():
    out=[]
    if len(v):
        budget=num(v.get("Budget_Amount",pd.Series(dtype=float)).sum()); actual=num(v.get("Actual_Amount",pd.Series(dtype=float)).sum()); variance=num(v.get("Variance",pd.Series(dtype=float)).sum())
        unf=num(v.loc[v["Variance"]>0,"Variance"].sum()) if "Variance" in v else 0
        fav=abs(num(v.loc[v["Variance"]<0,"Variance"].sum())) if "Variance" in v else 0
        out.append(("FINANCIAL POSITION", f"Net variance is {money(variance)} unfavorable, with {money(unf)} gross unfavorable exposure partially offset by {money(fav)} favorable variance."))
    if len(r):
        imp=safe_col(r,["Unfavorable_Variance","Unfavorable","Exposure"])
        if imp:
            rr=r.sort_values(imp,ascending=False).iloc[0]
            out.append(("TOP RECURRING DRIVER", f"{rr.get('Department','—')} / {rr.get('Category','—')} is the largest recurring unfavorable driver at {money(rr.get(imp,0))}."))
    if len(w) and "Early_Warning" in w.columns:
        red=int((w["Early_Warning"].astype(str).str.upper()=="RED").sum()); amb=int((w["Early_Warning"].astype(str).str.upper()=="AMBER").sum())
        out.append(("FORWARD RISK", f"{red} forecast lines are RED and {amb} are AMBER; these should be reviewed before the period closes."))
    ex=int(planning_summary.get("Exception Transactions",0))
    if ex:
        out.append(("DATA INTEGRITY", f"{ex} planning exceptions remain outside mapped budget lines and should not be forced into the model."))
    return out[:4]


def insight_cards():
    cards=[]
    for title,textv in management_insights():
        cards.append(f"<div class='card soft-card'><div class='kicker'>{esc(title)}</div><div class='sub' style='font-size:12px;color:#4f596c'>{esc(textv)}</div></div>")
    return cards

# ============================================================
# EXECUTIVE
# ============================================================
if page == "Executive":
    b = num(v.get("Budget_Amount", pd.Series(dtype=float)).sum()); a = num(v.get("Actual_Amount", pd.Series(dtype=float)).sum()); var = num(v.get("Variance", pd.Series(dtype=float)).sum())
    unf = num(v.loc[v["Variance"]>0,"Variance"].sum()) if "Variance" in v else 0
    fav = abs(num(v.loc[v["Variance"]<0,"Variance"].sum())) if "Variance" in v else 0
    mat = int(m["Is_Material"].sum()) if "Is_Material" in m else 0
    red = int((w["Early_Warning"].astype(str).str.upper()=="RED").sum()) if "Early_Warning" in w else 0
    hp = int(q["Decision_Priority"].astype(str).str.upper().isin(["HIGH PRIORITY","CRITICAL"]).sum()) if "Decision_Priority" in q else 0

    c1,c2,c3 = st.columns([1.18,1,1])
    with c1: st.markdown(card("NET VARIANCE", money(var), f"{pct(var/b*100 if b else 0)} of budget • {money(unf)} unfavorable exposure", "card pink-card"), unsafe_allow_html=True)
    with c2: st.markdown(card("ACTUAL SPEND", money(a), f"Budget {money(b)} • utilization {pct(a/b*100 if b else 0)}", "card"), unsafe_allow_html=True)
    with c3: st.markdown(card("WHAT NEEDS ATTENTION", f"{hp:,}", f"{red} RED forecast lines • {mat:,} material combinations", "card blue-card"), unsafe_allow_html=True)

    section("Executive snapshot","The four numbers a CFO should see first")
    metric_grid([
        ("Unfavorable exposure", money(unf), "gross unfavorable variance"),
        ("Favorable offset", money(fav), "gross favorable variance"),
        ("Materiality rate", pct(mat/len(v)*100 if len(v) else 0), "of planning combinations"),
        ("RED alerts", f"{red:,}", "early-warning records"),
        ("High priority", f"{hp:,}", "CFO decisions"),
        ("Planning exceptions", f"{int(planning_summary.get('Exception Transactions',0)):,}", "explicitly unresolved"),
    ])

    c1,c2 = st.columns([1.25,.75])
    with c1:
        if len(monthly):
            fig=go.Figure(); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Budget,name="Budget",mode="lines",line=dict(color=INK,width=2))); fig.add_trace(go.Scatter(x=monthly.Month,y=monthly.Actual,name="Actual",mode="lines",line=dict(color=PINK,width=3))); chart("Budget vs actual","Monthly performance trajectory",fig,340,"ex_trend")
    with c2:
        if len(v):
            split=pd.DataFrame({"Type":["Unfavorable","Favorable"],"Amount":[unf,fav]}); fig=px.pie(split,names="Type",values="Amount",hole=.62,color="Type",color_discrete_map={"Unfavorable":PINK,"Favorable":GREEN}); chart("Exposure mix","Gross unfavorable versus favorable exposure",fig,340,"ex_mix")

    c1,c2 = st.columns([.95,1.05])
    with c1:
        dep=v.assign(Unfavorable=v["Variance"].clip(lower=0)).groupby("Department",as_index=False).Unfavorable.sum().sort_values("Unfavorable",ascending=False).head(7).sort_values("Unfavorable")
        if len(dep):
            fig=px.bar(dep,x="Unfavorable",y="Department",orientation="h",text="Unfavorable",color_discrete_sequence=[PINK]); fig.update_traces(texttemplate="₹%{x:,.0f}",textposition="outside"); chart("Top unfavorable departments","Where the biggest gross exposure sits",fig,335,"ex_dep")
    with c2:
        section("CFO activity","Top issues from the prioritized queue")
        issues=q.sort_values("Projected_Variance",ascending=False).head(5) if len(q) and "Projected_Variance" in q else q.head(5)
        for _,rr in issues.iterrows():
            risk_badge=status_html(rr.get("Risk_Band","—")) if str(rr.get("Risk_Band","—")) not in ["—","nan","None"] else status_html(rr.get("Decision_Priority","REVIEW"))
            st.markdown(f"<div class='issue'><div class='issue-top'><div><div class='issue-kicker'>{esc(rr.get('Business_Unit','—'))} • {esc(rr.get('Department','—'))}</div><div class='issue-name'>{esc(rr.get('Category','—'))}</div></div><div>{risk_badge}</div></div><div class='issue-metrics'><div><div class='lbl'>Current</div><div class='val'>{money(verified_current_variance(rr, v))}</div></div><div><div class='lbl'>Projected</div><div class='val'>{money(rr.get('Projected_Variance',0))}</div></div><div><div class='lbl'>Risk</div><div class='val'>{num(rr.get('Risk_Score',0)):.0f}</div></div></div></div>", unsafe_allow_html=True)

    section("What matters","Management interpretation from verified deterministic analytics")
    top_driver = None
    if len(r):
        imp = "Unfavorable_Variance" if "Unfavorable_Variance" in r.columns else None
        top_driver = r.sort_values(imp, ascending=False).iloc[0] if imp else r.iloc[0]
    driver_text = f"{top_driver.get('Department','—')} / {top_driver.get('Category','—')} contributes {money(top_driver.get('Unfavorable_Variance',0))} of recurring unfavorable exposure." if top_driver is not None else "No recurring driver is available for the selected scope."
    st.markdown(f"<div class='insight-box'><b>Management focus:</b> {esc(driver_text)} <br><b>Data integrity:</b> {int(planning_summary.get('Exception Transactions',0))} planning exceptions remain explicitly outside mapped budget lines. <br><b>Forward risk:</b> {red} RED early-warning records require attention before the projected period closes.</div>", unsafe_allow_html=True)

# ============================================================
# KPI
# ============================================================
elif page == "KPI Cockpit":
    b=num(v.get("Budget_Amount",pd.Series(dtype=float)).sum()); a=num(v.get("Actual_Amount",pd.Series(dtype=float)).sum()); var=num(v.get("Variance",pd.Series(dtype=float)).sum()); mat=int(m["Is_Material"].sum()) if "Is_Material" in m else 0; red=int((w["Early_Warning"].astype(str).str.upper()=="RED").sum()) if "Early_Warning" in w else 0; hp=int(q["Decision_Priority"].astype(str).str.upper().isin(["HIGH PRIORITY","CRITICAL"]).sum()) if "Decision_Priority" in q else 0
    metric_grid([("Budget",money(b),"planned spend"),("Actual",money(a),"matched actuals"),("Net variance",money(var),pct(var/b*100 if b else 0)),("Material",f"{mat:,}","material combinations"),("RED alerts",f"{red:,}","immediate attention"),("High priority",f"{hp:,}","CFO queue")])
    c1,c2=st.columns([1.18,.82])
    with c1:
        if len(monthly):
            fig=go.Figure(go.Bar(x=monthly.Month,y=monthly.Variance,marker_color=[PINK if z>=0 else GREEN for z in monthly.Variance],hovertemplate="%{x|%b %Y}<br>Variance=%{y:,.0f}<extra></extra>")); chart("Monthly variance columns","Positive = unfavorable • negative = favorable",fig,360,"kpi_cols")
    with c2:
        split=pd.DataFrame({"Type":["Unfavorable","Favorable"],"Amount":[num(v.loc[v.Variance>0,"Variance"].sum()),abs(num(v.loc[v.Variance<0,"Variance"].sum()))]}) if len(v) and "Variance" in v else pd.DataFrame()
        if len(split):
            fig=px.pie(split,names="Type",values="Amount",hole=.62,color="Type",color_discrete_map={"Unfavorable":PINK,"Favorable":GREEN}); chart("Favorable vs unfavorable","Gross exposure split",fig,360,"kpi_pie")
    c1,c2=st.columns([.9,1.1])
    with c1:
        if "Variance_Direction" in v.columns:
            d=v["Variance_Direction"].value_counts().rename_axis("Direction").reset_index(name="Records"); fig=px.bar(d,x="Direction",y="Records",color="Direction",color_discrete_map={"Unfavorable":PINK,"Favorable":GREEN},text="Records"); chart("Variance direction","Number of planning combinations by direction",fig,320,"kpi_dir")
    with c2:
        if len(monthly):
            roll=monthly.copy(); roll["3M Avg"]=roll.Variance.rolling(3,min_periods=1).mean(); fig=go.Figure(); fig.add_trace(go.Scatter(x=roll.Month,y=roll.Variance,name="Monthly",mode="lines+markers",line=dict(color=PINK,width=2.7))); fig.add_trace(go.Scatter(x=roll.Month,y=roll["3M Avg"],name="3M Avg",mode="lines",line=dict(color=LAVENDER,width=2,dash="dot"))); chart("Variance trend","Monthly variance versus smoothed trajectory",fig,320,"kpi_trend")
    section("KPI interpretation","What these indicators mean for management")
    st.markdown(f"<div class='insight-box'>Actual spend is <b>{pct(a/b*100 if b else 0)}</b> of budget. The net position is <b>{money(var)}</b> unfavorable, while <b>{money(num(v.loc[v.Variance>0,'Variance'].sum()) if len(v) else 0)}</b> of unfavorable exposure is partially offset by favorable variance. {red} RED forecast combinations and {hp} high-priority queue records indicate where attention is required.</div>", unsafe_allow_html=True)

# ============================================================
# ANALYTICS
# ============================================================
elif page == "Analytics":
    if len(v)==0:
        st.info("No records match the current filters.")
    else:
        cat=v.assign(Unfavorable=v["Variance"].clip(lower=0)).groupby("Category",as_index=False).Unfavorable.sum().sort_values("Unfavorable",ascending=False).head(10).sort_values("Unfavorable")
        c1,c2=st.columns([1.02,.98])
        with c1:
            cat["Label"]=cat["Unfavorable"].apply(money); fig=px.bar(cat,x="Unfavorable",y="Category",orientation="h",text="Label",color_discrete_sequence=[PINK]); fig.update_traces(textposition="outside",hovertemplate="%{y}<br>Unfavorable: %{x:,.0f}<extra></extra>"); chart("Top unfavorable categories","Absolute unfavorable exposure",fig,360,"an_cat")
        with c2:
            fig=px.histogram(v,x="Variance",nbins=32,color_discrete_sequence=[INK]); chart("Variance distribution","Distribution across planning combinations",fig,360,"an_hist")
        c1,c2=st.columns([1.05,.95])
        with c1:
            heat=v.assign(Unfavorable=v["Variance"].clip(lower=0)).pivot_table(index="Department",columns="Category",values="Unfavorable",aggfunc="sum",fill_value=0)
            fig=go.Figure(go.Heatmap(z=heat.values,x=heat.columns,y=heat.index,colorscale=[[0,"#F7F4F6"],[.5,"#F7B4CA"],[1,PINK]],hovertemplate="Department=%{y}<br>Category=%{x}<br>Exposure=%{z:,.0f}<extra></extra>")); chart("Department × category heatmap","Financial concentration and hotspots",fig,430,"an_heat")
        with c2:
            p=cat.copy(); p["Cumulative %"]=((p.Unfavorable.cumsum()/p.Unfavorable.sum()*100) if p.Unfavorable.sum() else 0).clip(upper=100); p["Label"]=p["Unfavorable"].apply(money); fig=go.Figure(); fig.add_trace(go.Bar(x=p.Category,y=p.Unfavorable,text=p.Label,textposition="outside",name="Unfavorable",marker_color=PINK,hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>")); fig.add_trace(go.Scatter(x=p.Category,y=p["Cumulative %"],name="Cumulative %",yaxis="y2",mode="lines+markers",line=dict(color=INK,width=2.3),hovertemplate="Cumulative: %{y:.1f}%<extra></extra>")); fig.update_layout(yaxis2=dict(overlaying="y",side="right",range=[0,100],ticksuffix="%",title="Cumulative %")); chart("Pareto of unfavorable exposure","Cumulative share across the displayed top categories",fig,430,"an_pareto")
        section("Business unit concentration","Which business units carry the category hotspots?")
        heat2=v.assign(Unfavorable=v["Variance"].clip(lower=0)).pivot_table(index="Business_Unit",columns="Category",values="Unfavorable",aggfunc="sum",fill_value=0)
        fig=go.Figure(go.Heatmap(z=heat2.values,x=heat2.columns,y=heat2.index,colorscale=[[0,"#F7F4F6"],[.5,"#F8C7D6"],[1,PINK]],hovertemplate="BU=%{y}<br>Category=%{x}<br>Exposure=%{z:,.0f}<extra></extra>")); chart("Business unit × category heatmap","Where categories are creating exposure by BU",fig,410,"an_bu_heat")

# ============================================================
# STATISTICS
# ============================================================
elif page == "Statistics":
    vals=v["Variance"].dropna() if "Variance" in v.columns else pd.Series(dtype=float)
    mean=vals.mean() if len(vals) else 0; med=vals.median() if len(vals) else 0; sd=vals.std() if len(vals) else 0; cv=(sd/abs(mean)*100) if mean else 0; p95=vals.quantile(.95) if len(vals) else 0; outliers=int(((vals-mean).abs()>2*sd).sum()) if len(vals) and sd else 0
    metric_grid([("Mean",money(mean),"average planning combination"),("Median",money(med),"middle planning combination"),("Std. dev.",money(sd),"dispersion"),("CV",pct(cv),"relative volatility"),("P95",money(p95),"95th percentile"),("Outliers",f"{outliers:,}","|z| > 2 proxy")])
    c1,c2=st.columns(2)
    with c1:
        fig=px.histogram(pd.DataFrame({"Variance":vals}),x="Variance",nbins=36,color_discrete_sequence=[INK]); chart("Variance histogram","Shape and spread of variance",fig,390,"st_hist")
    with c2:
        fig=go.Figure(go.Box(y=vals,boxpoints="outliers",marker_color=PINK,line_color=PINK,fillcolor=PINK_SOFT)); chart("Variance box plot","Central spread and extreme observations",fig,390,"st_box")
    if len(v) and "Department" in v.columns:
        dep=v.groupby("Department",as_index=False).agg(Mean_Variance=("Variance","mean"),Volatility=("Variance","std"),Records=("Variance","size")).fillna(0)
        fig=px.scatter(dep,x="Mean_Variance",y="Volatility",size="Records",text="Department",color_discrete_sequence=[PINK]); fig.update_traces(textposition="top center"); chart("Department mean vs volatility","Higher = more volatile planning performance",fig,380,"st_scatter")
    if len(monthly):
        dd=monthly.copy(); dd["3M Avg"]=dd.Variance.rolling(3,min_periods=1).mean(); fig=go.Figure(); fig.add_trace(go.Scatter(x=dd.Month,y=dd.Variance,name="Variance",mode="lines+markers",line=dict(color=PINK,width=2.5))); fig.add_trace(go.Scatter(x=dd.Month,y=dd["3M Avg"],name="3M Avg",mode="lines",line=dict(color=LAVENDER,width=2,dash="dot"))); chart("Statistical trend","Variance over time with smoothed central tendency",fig,340,"st_trend")

# ============================================================
# FORECAST & RISK
# ============================================================
elif page == "Forecast & Risk":
    proj=num(f["Projected_Variance"].clip(lower=0).sum()) if len(f) and "Projected_Variance" in f else 0
    reds=int((w["Early_Warning"].astype(str).str.upper()=="RED").sum()) if len(w) and "Early_Warning" in w else 0
    amb=int((w["Early_Warning"].astype(str).str.upper()=="AMBER").sum()) if len(w) and "Early_Warning" in w else 0
    grn=int((w["Early_Warning"].astype(str).str.upper()=="GREEN").sum()) if len(w) and "Early_Warning" in w else 0
    crit=int((rsk["Risk_Band"].astype(str).str.upper()=="CRITICAL").sum()) if len(rsk) and "Risk_Band" in rsk else 0
    metric_grid([("Projected unfavorable",money(proj),"forecast exposure"),("RED alerts",f"{reds:,}","immediate attention"),("AMBER alerts",f"{amb:,}","management review"),("GREEN",f"{grn:,}","monitor"),("Critical risks",f"{crit:,}","highest risk band"),("Forecast lines",f"{len(f):,}","planning combinations")])
    if len(monthly):
        x=monthly.copy(); x["3M Run Rate"]=x.Actual.rolling(3,min_periods=1).mean(); last=x.Month.max(); fut=pd.date_range(last+pd.offsets.MonthBegin(1),periods=6,freq="MS"); base=float(x["3M Run Rate"].iloc[-1]) if len(x) else 0; ext=pd.DataFrame({"Month":fut,"Forecast": [base]*len(fut)}); fig=go.Figure(); fig.add_trace(go.Scatter(x=x.Month,y=x.Budget,name="Budget",mode="lines",line=dict(color=INK,width=2))); fig.add_trace(go.Scatter(x=x.Month,y=x.Actual,name="Actual",mode="lines",line=dict(color=PINK,width=3))); fig.add_trace(go.Scatter(x=x.Month,y=x["3M Run Rate"],name="3M run rate",mode="lines",line=dict(color=LAVENDER,width=2,dash="dot"))); fig.add_trace(go.Scatter(x=ext.Month,y=ext.Forecast,name="Forecast",mode="lines+markers",line=dict(color=LAVENDER,width=3,dash="dash"))); fig.add_vline(x=last,line_dash="dot",line_color="#B8B1EA",annotation_text="Forecast starts"); fig.update_xaxes(type="date",tickformat="%b %Y",dtick="M6",showgrid=False); fig.update_yaxes(tickformat="~s"); chart("Budget vs actual vs forecast","Actuals, budget and a transparent six-month run-rate extension",fig,390,"fr_line")
    c1,c2=st.columns([1.0,1.0])
    with c1:
        if len(rsk) and "Risk_Score" in rsk.columns and "Projected_Variance" in rsk.columns:
            rr=rsk.copy(); rr["Risk_Band"]=rr.get("Risk_Band",pd.Series("LOW",index=rr.index)); fig=px.scatter(rr,x="Risk_Score",y="Projected_Variance",color="Risk_Band",hover_data=[c for c in ["Business_Unit","Department","Category"] if c in rr.columns],color_discrete_map={"CRITICAL":RED,"HIGH":PINK,"MODERATE":AMBER,"LOW":GREEN}); fig.update_traces(marker_size=12); chart("Risk vs projected exposure","Further right and higher = greater priority",fig,350,"fr_scatter")
    with c2:
        if len(w):
            ww=w["Early_Warning"].astype(str).str.upper().value_counts().rename_axis("Band").reset_index(name="Records"); fig=px.pie(ww,names="Band",values="Records",hole=.62,color="Band",color_discrete_map={"RED":RED,"AMBER":AMBER,"GREEN":GREEN}); chart("Early-warning mix","Forward status of forecast lines",fig,350,"fr_pie")
    if len(f) and "Projected_Variance" in f.columns:
        top=f.sort_values("Projected_Variance",ascending=False).head(10).copy(); fig=px.bar(top.sort_values("Projected_Variance"),x="Projected_Variance",y="Category",color="Business_Unit" if "Business_Unit" in top.columns else None,orientation="h",color_discrete_sequence=[PINK]); chart("Top projected exposures","Highest projected variance lines",fig,365,"fr_bar")

# ============================================================
# SCENARIOS
# ============================================================
elif page == "Scenarios":
    rec_name, basev, scenv, improvement = fmt_scenario_summary(scenarios, scenario_summary)
    st.markdown(f"<div class='card blue-card'><div class='kicker'>RECOMMENDED SCENARIO</div><div class='card-value' style='font-size:36px'>{esc(rec_name)}</div><div class='sub'>Modeled recommendation • validate feasibility, service impact and contractual constraints before adoption</div></div>", unsafe_allow_html=True)
    metric_grid([("Base projected variance",money(basev),"current modeled path"),("Modeled improvement",money(improvement),"versus base"),("Scenario projected variance",money(scenv),"after intervention"),("Scenario options",f"{len(scenarios):,}","compared cases"),("Projected records",f"{len(f):,}","forecast lines"),("Control","Modeled","not a blanket cut")])
    if len(scenarios):
        c1,c2=st.columns([1.05,.95])
        with c1:
            plot=scenarios.sort_values("Variance_Improvement").copy(); plot["Label"]=plot["Variance_Improvement"].apply(money); fig=px.bar(plot,x="Variance_Improvement",y="Scenario_Name",orientation="h",text="Label",color_discrete_sequence=[PINK]); fig.update_traces(textposition="outside",hovertemplate="%{y}<br>Improvement: %{x:,.0f}<extra></extra>"); chart("Scenario ranking","Improvement versus base",fig,350,"sc_rank")
        with c2:
            long=scenarios[[c for c in ["Scenario_Name","Base_Projected_Variance","Scenario_Projected_Variance"] if c in scenarios.columns]].melt(id_vars="Scenario_Name",var_name="Metric",value_name="Value"); fig=px.bar(long,x="Scenario_Name",y="Value",color="Metric",barmode="group",color_discrete_map={"Base_Projected_Variance":INK,"Scenario_Projected_Variance":PINK}); chart("Base vs intervention","Projected variance by scenario",fig,350,"sc_group")
        section("Scenario decision cards","The modeled outcomes before detailed evidence")
        cards=[]
        for _,sr in scenarios.sort_values("Variance_Improvement",ascending=False).iterrows():
            cards.append(f"<div class='card soft-card'><div class='kicker'>{esc(sr.get('Scenario_Name','Scenario'))}</div><div class='card-value sm'>{money(sr.get('Scenario_Projected_Variance',0))}</div><div class='sub'>Improvement {money(sr.get('Variance_Improvement',0))} • Savings {money(sr.get('Forecast_Savings',0))}</div></div>")
        scols=st.columns(min(4,len(cards)))
        for c,bx in zip(scols,cards):
            with c: st.markdown(bx,unsafe_allow_html=True)
        section("Detailed scenario evidence","Exact values remain available below for audit and verification")
        disp=[c for c in ["Scenario_Name","Base_Projected_Variance","Scenario_Projected_Variance","Variance_Improvement","Forecast_Savings","Budget_Headroom_Change","Projected_Overspend_Records","Projected_Underspend_Records"] if c in scenarios.columns]
        shown=scenarios[disp].copy()
        for c in shown.columns:
            if c!="Scenario_Name": shown[c]=shown[c].apply(lambda z: money(z) if "Variance" in c or "Savings" in c or "Headroom" in c else (f"{num(z):,.0f}" if "Records" in c else z))
        st.dataframe(shown,width="stretch",hide_index=True,height=250)
    st.markdown("<div class='insight-box'><b>Scenario control:</b> scenario outputs are mathematical interventions, not unconditional instructions. Validate feasibility, service impact, contractual constraints and operational capacity before adoption.</div>",unsafe_allow_html=True)

# ============================================================
# CFO QUEUE
# ============================================================
elif page == "CFO Queue":
    hp=int(q["Decision_Priority"].astype(str).str.upper().isin(["HIGH PRIORITY","CRITICAL"]).sum()) if len(q) and "Decision_Priority" in q else 0
    rv=int((q["Decision_Priority"].astype(str).str.upper()=="REVIEW").sum()) if len(q) and "Decision_Priority" in q else 0
    mo=int((q["Decision_Priority"].astype(str).str.upper()=="MONITOR").sum()) if len(q) and "Decision_Priority" in q else 0
    cur_exp=num(pd.to_numeric(q.get("Current_Variance", pd.Series(dtype=float)), errors="coerce").clip(lower=0).sum()) if len(q) and "Current_Variance" in q else 0
    proj_exp=num(q["Projected_Variance"].clip(lower=0).sum()) if len(q) and "Projected_Variance" in q else 0
    metric_grid([("High priority",f"{hp:,}","immediate decisions"),("Review",f"{rv:,}","management review"),("Monitor",f"{mo:,}","watch trajectory"),("Queue records",f"{len(q):,}","planning combinations"),("Current exposure",money(cur_exp),"unfavorable"),("Projected exposure",money(proj_exp),"future unfavorable")])
    section("Decision queue","Each card is a management action, not just a variance row")
    issues=q.sort_values("Projected_Variance",ascending=False).head(12) if len(q) and "Projected_Variance" in q.columns else q.head(12)
    for start in range(0,len(issues),2):
        pair=st.columns(2)
        for col,(_,rr) in zip(pair, issues.iloc[start:start+2].iterrows()):
            with col:
                decision=rr.get("Decision_Priority","REVIEW")
                rb=rr.get("Risk_Band","—"); rs=num(rr.get("Risk_Score",0))
                st.markdown(f"<div class='issue'><div class='issue-top'><div><div class='issue-kicker'>{esc(rr.get('Business_Unit','—'))} • {esc(rr.get('Department','—'))}</div><div class='issue-name'>{esc(rr.get('Category','—'))}</div></div><div>{status_html(decision)}</div></div><div class='issue-metrics'><div><div class='lbl'>Current variance</div><div class='val'>{money(verified_current_variance(rr, v))}</div></div><div><div class='lbl'>Projected variance</div><div class='val'>{money(rr.get('Projected_Variance',0))}</div></div><div><div class='lbl'>Risk</div><div class='val'>{esc(rb) if rb not in ["—","nan","None"] else '—'}{f" · {rs:.0f}" if rs else ""}</div></div></div><div class='issue-action'><b>Recommended action:</b> {esc(rr.get('Recommended_Action',rr.get('Management_Action','Investigate the variance and validate the driver.')))}</div></div>",unsafe_allow_html=True)
    download(q,"Download CFO queue","varia_cfo_queue.csv")

# ============================================================
# AI INSIGHTS
# ============================================================
elif page == "AI Insights":
    section("Verified management position","Deterministic insights are shown first; AI only communicates verified evidence")
    ic=insight_cards()
    if ic:
        cc=st.columns(min(4,len(ic)))
        for col,box in zip(cc,ic):
            with col: st.markdown(box,unsafe_allow_html=True)
    c1,c2=st.columns([1.02,.98])
    with c1: st.markdown(card("LOCAL MODEL","Qwen 2.5 7B","Local commentary engine; VARIA validates output before it is treated as grounded","card blue-card"),unsafe_allow_html=True)
    with c2: st.markdown("<div class='card'><div class='kicker'>AI GOVERNANCE</div><div class='card-title'>VARIA decides WHAT. Qwen decides HOW.</div><div class='sub' style='font-size:12px'>Deterministic financial calculations remain authoritative. The local model is constrained by schema, numeric, financial-semantic, evidence, causal and trend controls.</div></div>",unsafe_allow_html=True)
    if st.button("✦ Generate grounded CFO briefing",type="primary",width="content",key="generate_ai"):
        try:
            client=OllamaLLMEngine(model="qwen2.5:7b",timeout=180)
            engine=VARIAGroundedAIEngine(llm_client=client,max_retries=2)
            st.session_state.ai_result=engine.generate(context_payload=evidence_pack if isinstance(evidence_pack,dict) else outputs.get("llm_context",{}),temperature=0.0)
        except Exception as exc:
            st.session_state.ai_result={"status":"REVIEW","grounded":False,"error":str(exc),"validation":{}}
        st.rerun()
    ai_res=st.session_state.get("ai_result")
    if isinstance(ai_res,dict):
        val=ai_res.get("validation",{}) or {}
        checks=[("Schema",bool(val.get("schema_ok",False))), ("Numeric",bool(val.get("numeric_grounding",False))), ("Financial semantics",bool(val.get("financial_semantics",False))), ("Evidence",bool(val.get("evidence_coverage",False))), ("Causal",bool(val.get("causal_control",False))), ("Trend",bool(val.get("trend_control",False)))]
        blocks=[]
        for lab,ok in checks:
            blocks.append(f"<div class='ai-check {'pass' if ok else 'fail'}'><div class='kicker'>{esc(lab)}</div><div class='v'>{'✓' if ok else '!'}</div><div class='sub'>{'PASS' if ok else 'REVIEW'}</div></div>")
        st.markdown(f"<div class='ai-grid'>{''.join(blocks)}</div>",unsafe_allow_html=True)
        if ai_res.get("status")=="PASS" and ai_res.get("grounded") is True:
            st.markdown("<div class='card green-card' style='margin-top:12px'><div class='kicker'>GROUNDED STATUS</div><div class='card-title'>AI commentary passed all active controls</div><div class='sub'>The generated narrative is safe to display alongside deterministic VARIA outputs.</div></div>",unsafe_allow_html=True)
        else:
            st.markdown("<div class='card amber-card' style='margin-top:12px'><div class='kicker'>REVIEW STATUS</div><div class='card-title'>AI commentary is held for review</div><div class='sub'>The deterministic VARIA analysis remains fully usable. No AI statement should override the verified financial model.</div></div>",unsafe_allow_html=True)
        structured=ai_res.get("structured")
        if isinstance(structured,dict):
            eo=structured.get("executive_observation",{})
            txt=eo.get("text",eo) if isinstance(eo,dict) else str(eo)
            st.markdown(f"<div class='card' style='margin-top:12px'><div class='kicker'>EXECUTIVE OBSERVATION</div><div class='card-title'>{esc(txt)}</div></div>",unsafe_allow_html=True)
            section("Key issues","Every statement is associated with evidence IDs")
            for item in (structured.get("key_issues",[]) or [])[:8]:
                t=item.get("text","") if isinstance(item,dict) else str(item); ids=item.get("evidence_ids",[]) if isinstance(item,dict) else []
                st.markdown(f"<div class='evidence-row'><div class='evidence-id'>{esc(' • '.join(ids))}</div><div class='evidence-text'>{esc(t)}</div></div>",unsafe_allow_html=True)
            if structured.get("evidence_gaps"):
                st.markdown(f"<div class='insight-box' style='margin-top:10px'><b>Evidence gaps:</b> {esc(' • '.join(structured.get('evidence_gaps',[])))}</div>",unsafe_allow_html=True)
        if ai_res.get("error"):
            st.error(ai_res["error"])
    section("Management recommendations","Deterministic actions remain available even when the LLM returns REVIEW")
    recs = recommendations if isinstance(recommendations,pd.DataFrame) else pd.DataFrame()
    if len(recs):
        for _,rr in recommendation_rows(recs,4):
            st.markdown(f"<div class='evidence-row'><div class='evidence-id'>{esc(priority_text(rr))} • {esc(rr.get('Business_Unit','—'))} • {esc(rr.get('Department','—'))}</div><div class='evidence-text'><b>{esc(rr.get('Category','—'))}</b> — {esc(rec_text(rr))}</div></div>",unsafe_allow_html=True)
    section("Verified financial feed","The AI page still shows deterministic intelligence when the LLM is unavailable")
    if len(ai_base):
        show=ai_base.copy()
        if "Projected_Variance" in show: show=show.sort_values("Projected_Variance",ascending=False).head(12)
        st.dataframe(show,width="stretch",hide_index=True,height=320)

# ============================================================
# RECOMMENDATIONS
# ============================================================
elif page == "Recommendations":
    section("Management recommendations","Prioritized actions generated from verified VARIA analytics")
    recs = recommendations.copy() if isinstance(recommendations,pd.DataFrame) else pd.DataFrame()
    # Build a deterministic fallback when the recommendation engine returns no rows.
    if recs.empty and len(q):
        recs=q.copy()
    if not recs.empty:
        proj_col=safe_col(recs,["Projected_Variance","Projected_Exposure","Forecast_Variance"])
        priority_col=safe_col(recs,["Priority","Decision_Priority","Risk_Band"])
        metric_grid([
            ("Recommendation records",f"{len(recs):,}","decision-ready rows"),
            ("High priority",f"{sum(str(rr.get('Decision_Priority', rr.get('Priority', ''))).upper()=='HIGH PRIORITY' for _,rr in recs.iterrows()):,}","immediate attention"),
            ("Projected exposure",money(recs[proj_col].clip(lower=0).sum()) if proj_col else "—","future unfavorable"),
            ("Action engine", "ACTIVE", "deterministic layer"),
            ("AI override", "NONE", "VARIA remains authoritative"),
            ("Evidence", f"{len(evidence_pack.get('evidence_ledger',[])):,}" if isinstance(evidence_pack,dict) else "—", "verified evidence records"),
        ])
        for start in range(0,min(len(recs),10),2):
            cols=st.columns(2)
            for col,(_,rr) in zip(cols,recs.iloc[start:start+2].iterrows()):
                with col:
                    pr = str(rr.get("Decision_Priority", rr.get("Priority", "REVIEW")))
                    rb = str(rr.get("Risk_Band", "—"))
                    rs = num(rr.get("Risk_Score", 0))
                    current = verified_current_variance(rr, v)
                    projected = rr.get(proj_col,0) if proj_col else 0
                    risk_label = (f"{rb} · {rs:.0f}" if rb not in ["—","nan","None",""] else (f"{rs:.0f}" if rs else "—"))
                    st.markdown(f"<div class='issue'><div class='issue-top'><div><div class='issue-kicker'>{esc(rr.get('Business_Unit','—'))} • {esc(rr.get('Department','—'))}</div><div class='issue-name'>{esc(rr.get('Category','—'))}</div></div>{status_html(pr)}</div><div class='issue-metrics'><div><div class='lbl'>Current variance</div><div class='val'>{money(current)}</div></div><div><div class='lbl'>Projected variance</div><div class='val'>{money(projected)}</div></div><div><div class='lbl'>Risk</div><div class='val' style='font-size:13px'>{esc(risk_label)}</div></div></div><div class='issue-action'><b>Recommended action:</b> {esc(rec_text(rr))}</div></div>",unsafe_allow_html=True)
        section("Recommendation evidence","Raw recommendation output is available for audit")
        st.dataframe(recs.head(25),width="stretch",hide_index=True,height=310)
    else:
        st.success("No recommendations are available for the selected scope.")
    section("Management principles","How VARIA prevents recommendations from becoming unsafe blanket instructions")
    st.markdown("<div class='insight-box'><b>1.</b> Investigate the largest verified drivers first. &nbsp; <b>2.</b> Validate business need and operational constraints. &nbsp; <b>3.</b> Use scenario analysis before committing resources. &nbsp; <b>4.</b> Treat AI commentary as communication, never as an override of deterministic financial calculations.</div>",unsafe_allow_html=True)

# ============================================================
# ROOT CAUSE
# ============================================================
elif page == "Root Cause":
    if len(r):
        imp="Unfavorable_Variance" if "Unfavorable_Variance" in r.columns else None
        name="Category" if "Category" in r.columns else ("Department" if "Department" in r.columns else r.columns[0])
        drv=r.sort_values(imp,ascending=False) if imp else r
        top=drv.head(10).sort_values(imp) if imp else drv.head(10).sort_values(name)
        fig=px.bar(top,x=imp,y=name,orientation="h",text=imp,color_discrete_sequence=[PINK]) if imp else px.bar(top,x=name,orientation="h",color_discrete_sequence=[PINK]);
        if imp:
            top["Label"] = top[imp].apply(money)
            fig.update_traces(text=top["Label"],textposition="outside")
        chart("Recurring driver exposure","Highest evidence-based recurring unfavorable drivers",fig,380,"rc_driver")
        c1,c2=st.columns([1.05,.95])
        with c1:
            dep=v.assign(Unfavorable=v["Variance"].clip(lower=0)).groupby(["Department","Category"],as_index=False).agg(Unfavorable=("Unfavorable","sum"),Occurrences=("Variance","size")).sort_values("Unfavorable",ascending=False).head(12)
            fig=px.bar(dep.sort_values("Unfavorable"),x="Unfavorable",y="Category",color="Department",orientation="h",color_discrete_sequence=[PINK,LAVENDER,INK,AMBER,GREEN]); chart("Driver mix","Department × category combinations driving exposure",fig,390,"rc_mix")
        with c2:
            st.markdown("<div class='card blue-card'><div class='kicker'>INTERPRETATION CONTROL</div><div class='card-title'>Pattern ≠ causality</div><div class='sub' style='font-size:12px'>VARIA can establish recurrence, concentration, frequency, materiality and direction. A causal explanation is displayed only when supporting evidence exists.</div></div>",unsafe_allow_html=True)
            sel=st.selectbox("Driver drill-down",drv[name].astype(str).tolist()[:20],key="driver_pick")
            drill=v[v[name].astype(str)==sel] if name in v.columns else v
            dd=safe_dt(drill,"Month").groupby("Month",as_index=False).agg(Variance=("Variance","sum"))
            if len(dd):
                fig=px.line(dd,x="Month",y="Variance",markers=True,color_discrete_sequence=[PINK]); chart("Selected driver trend","Monthly movement for the chosen recurring driver",fig,315,"rc_trend")
        st.markdown("<div style='margin-top:10px'></div>",unsafe_allow_html=True)
        show=r.head(12).copy()
        for c in ["Unfavorable_Variance","Average_Variance"]:
            if c in show.columns: show[c]=show[c].apply(money)
        if "Average_Variance_Pct" in show.columns: show["Average_Variance_Pct"]=show["Average_Variance_Pct"].apply(pct)
        st.dataframe(show,width="stretch",hide_index=True,height=250)
    else:
        st.info("No recurring drivers are available for the selected scope.")

# ============================================================
# DATA & AUDIT
# ============================================================
elif page == "Data & Audit":
    raw=num(actuals.get("Actual_Amount",pd.Series(dtype=float)).sum()) if isinstance(actuals,pd.DataFrame) else 0
    matched=num(v.get("Actual_Amount",pd.Series(dtype=float)).sum()) if len(v) else 0
    ex=int(planning_summary.get("Exception Transactions",0)); examt=num(planning_summary.get("Exception Amount",raw-matched)); mr=num(planning_summary.get("Matched Rate %",matched/raw*100 if raw else 0))
    metric_grid([("Budget rows",f"{len(budget):,}","planning combinations"),("Actual transactions",f"{len(actuals):,}","source rows"),("Matched actuals",f"{len(outputs.get('matched_actuals',pd.DataFrame())):,}","reconciled"),("Exceptions",f"{ex:,}",money(examt)+" exposure"),("Matched rate",pct(mr),"planning reconciliation"),("Raw actual amount",money(raw),"source total")])
    c1,c2=st.columns([1.05,.95])
    with c1:
        fig=go.Figure(go.Funnel(y=["Raw actuals","Matched actuals","Exception exposure"],x=[raw,matched,examt],textinfo="value",marker=dict(color=[INK,PINK,AMBER]))); fig.update_traces(texttemplate=[money(raw),money(matched),money(examt)]); chart("Reconciliation funnel","Raw → 99.97% matched → 0.03% exceptions; invalid matches are never silently forced",fig,345,"da_funnel")
    with c2:
        stages=int(num(audit_summary.get("Stages",0))); passes=int(num(audit_summary.get("Pass",0))); review=int(num(audit_summary.get("Review",0))); fail=int(num(audit_summary.get("Fail",0)))
        st.markdown(f"<div class='card blue-card'><div class='kicker'>AUDIT STATUS</div><div class='card-value'>{passes}/{stages}</div><div class='sub'>pipeline stages passing</div><div style='margin-top:12px'>{status_html('PASS' if fail==0 and review==0 and stages else 'REVIEW')}</div><div class='sub'>Review: {review} • Fail: {fail}</div></div>",unsafe_allow_html=True)
        st.markdown(f"<div class='insight-box' style='margin-top:10px'><b>Data quality status:</b> {'REVIEW' if ex else 'PASS'} — {ex} planning exceptions remain explicitly unresolved.</div>",unsafe_allow_html=True)
    section("Planning exceptions","The two exceptions are intentionally preserved for audit")
    if isinstance(planning_exceptions,pd.DataFrame) and len(planning_exceptions):
        st.dataframe(planning_exceptions,width="stretch",hide_index=True,height=260)
    else:
        st.success("No planning exceptions in this scope.")
    section("Audit lineage","Input → validation → analytics → decision intelligence")
    if len(audit): st.dataframe(audit,width="stretch",hide_index=True,height=250)
    section("Exports","Take verified VARIA outputs into management reporting")
    d1,d2,d3,d4=st.columns(4)
    with d1: download(v,"Variance CSV","varia_variance.csv")
    with d2: download(q,"CFO Queue CSV","varia_cfo_queue.csv")
    with d3: download(planning_exceptions,"Exceptions CSV","varia_exceptions.csv")
    with d4:
        st.download_button("CFO brief",str(cfo_report).encode(),"varia_cfo_brief.md","text/markdown",width="stretch")

st.markdown("<div style='text-align:center;color:#A0A7B4;font-size:9px;margin-top:34px'>◆ VARIA • deterministic financial analytics remain authoritative over generated language</div>", unsafe_allow_html=True)
