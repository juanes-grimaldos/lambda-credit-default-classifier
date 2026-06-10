"""
Credit Default Risk Dashboard — Portfolio Demo
Streamlit app that calls the deployed AWS Lambda endpoint.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scripts.simulate_values import values_simulation
from scripts.post import predict_payload
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Default Risk Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Base */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #0f172a;
    border-right: 1px solid #1e293b;
  }
  [data-testid="stSidebar"] * { color: #94a3b8 !important; }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
  [data-testid="stSidebar"] h3 { color: #f1f5f9 !important; }

  /* Main background */
  .main { background: #0f172a; }
  .block-container { padding-top: 1.5rem; }

  /* Cards */
  .risk-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
  }
  .risk-card-high { border-left: 4px solid #ef4444; }
  .risk-card-low  { border-left: 4px solid #22c55e; }

  /* Metric pills */
  .pill {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .pill-high { background: #450a0a; color: #fca5a5; }
  .pill-low  { background: #052e16; color: #86efac; }
  .pill-med  { background: #422006; color: #fdba74; }

  /* Probability bar container */
  .prob-bar-bg {
    background: #334155;
    border-radius: 4px;
    height: 8px;
    margin-top: 6px;
    overflow: hidden;
  }

  /* Typography */
  .section-title {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.75rem;
  }
  .mono { font-family: 'JetBrains Mono', monospace; }

  /* Override Streamlit metric */
  [data-testid="metric-container"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.75rem 1rem;
  }
  [data-testid="metric-container"] label { color: #64748b !important; font-size: 0.75rem !important; }
  [data-testid="metric-container"] [data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.4rem !important; }

  /* Table override */
  [data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

  /* Slider */
  [data-testid="stSlider"] .stSlider > div { color: #94a3b8; }

  /* Divider */
  hr { border-color: #1e293b; }
</style>
""", unsafe_allow_html=True)



SEX_MAP   = {1: "Male", 2: "Female"}
EDU_MAP   = {0: "Other", 1: "Graduate", 2: "University", 3: "High School", 4: "Other", 5: "Other", 6: "Other"}
MAR_MAP   = {0: "Other", 1: "Married", 2: "Single", 3: "Other"}

DEFAULT_THRESHOLD = 0.3367  # from opt_threshold.pkl (update if you retrain)




def risk_label(prob: float, threshold: float) -> tuple[str, str]:
    if prob >= threshold + 0.15:
        return "High Risk", "high"
    elif prob >= threshold:
        return "Default", "high"
    elif prob >= threshold - 0.1:
        return "Watch",   "med"
    else:
        return "Low Risk", "low"


def prob_bar_html(prob: float, threshold: float) -> str:
    pct = int(prob * 100)
    tpct = int(threshold * 100)
    color = "#ef4444" if prob >= threshold else "#22c55e"
    return f"""
    <div style="position:relative; margin-top:6px;">
      <div class="prob-bar-bg">
        <div style="width:{pct}%; height:8px; background:{color}; border-radius:4px; transition:width 0.4s;"></div>
      </div>
      <div style="position:absolute; top:-2px; left:{tpct}%; width:2px; height:12px;
                  background:#facc15; border-radius:1px;" title="Threshold"></div>
    </div>
    <div style="display:flex; justify-content:space-between; margin-top:3px;">
      <span style="font-size:0.68rem; color:#64748b; font-family:monospace;">{prob:.1%} probability</span>
      <span style="font-size:0.68rem; color:#64748b;">⬆ threshold {threshold:.2f}</span>
    </div>
    """
def call_lambda(df):
    try:
        payload = df.to_dict(orient="records")

        local_mode = os.getenv("LOCAL_LAMBDA", "false").lower() == "true"

        return predict_payload(
            payload=payload,
            local_mode=local_mode
        )

    except Exception as e:
        st.error(f"Inference failed: {e}")
        return None

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    n_clients = st.slider("Portfolio size", 10, 60, 30, 5)
    seed_val   = st.number_input("Random seed", value=42, step=1)

    st.markdown("### Decision threshold")
    threshold = st.slider(
        "Adjust threshold",
        min_value=0.10, max_value=0.70,
        value=DEFAULT_THRESHOLD, step=0.01,
        help="Yellow line on each bar. Lower = more conservative (flag more clients)."
    )
    st.caption(f"Model's optimized threshold: **{DEFAULT_THRESHOLD:.4f}** (PR-curve F1 max)")

    st.markdown("---")
    run_btn = st.button("🔄  Score portfolio", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### About")
    st.caption(
        "LightGBM classifier · UCI Default dataset · "
        "30k credit card holders · "
        "Threshold tuned via Precision-Recall curve · "
        "Deployed on AWS Lambda"
    )
    st.caption("Juanes — [GitHub](https://github.com)")


# ── Session state ──────────────────────────────────────────────────────────────

if "portfolio" not in st.session_state:
    st.session_state.portfolio = None
    st.session_state.results   = None

if run_btn or st.session_state.portfolio is None:
    with st.spinner("Generating synthetic clients and scoring via Lambda…"):
        sim = values_simulation(
                N=n_clients,
                seed=int(seed_val)
            )
        df = sim.generate()
        res = call_lambda(df)
        if res:
            st.session_state.portfolio = df
            st.session_state.results   = res

df  = st.session_state.portfolio
res = st.session_state.results

if df is None or res is None:
    st.stop()

probs     = np.array(res["probs"])
preds     = (probs >= threshold).astype(int)
opt_prob  = res.get("opt_prob", DEFAULT_THRESHOLD)

# ── Header ─────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="margin-bottom:1.5rem;">
  <p style="font-size:0.75rem; font-weight:600; letter-spacing:0.12em;
            text-transform:uppercase; color:#64748b; margin:0 0 4px 0;">
    ML INFERENCE · AWS LAMBDA
  </p>
  <h1 style="font-size:2rem; font-weight:700; color:black; margin:0; line-height:1.2;">
    Credit Default Risk Dashboard
  </h1>
  <p style="color:#94a3b8; margin:6px 0 0 0; font-size:0.9rem;">
    Synthetic portfolio of <strong style="color:#f1f5f9;">{n}</strong> clients
    scored in real-time via the deployed Lambda endpoint.
  </p>
</div>
""".format(n=len(df)), unsafe_allow_html=True)


# ── KPI row ────────────────────────────────────────────────────────────────────

n_flagged   = int(preds.sum())
n_safe      = len(preds) - n_flagged
avg_prob    = float(probs.mean())
flag_rate   = n_flagged / len(preds)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Portfolio size",    f"{len(df):,}")
c2.metric("Flagged (default)", f"{n_flagged}",   delta=f"{flag_rate:.0%} of portfolio",  delta_color="inverse")
c3.metric("Low risk",          f"{n_safe}")
c4.metric("Avg. default prob", f"{avg_prob:.1%}")

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)


# ── Distribution chart ────────────────────────────────────────────────────────

st.markdown('<p class="section-title">Probability distribution</p>', unsafe_allow_html=True)

buckets_low  = [0] * 10
buckets_high = [0] * 10
for p in probs:
    bucket = min(int(p * 10), 9)
    if (bucket / 10) >= threshold:
        buckets_high[bucket] += 1
    else:
        buckets_low[bucket] += 1

labels = [f"{i*10}–{(i+1)*10}%" for i in range(10)]
chart_df = pd.DataFrame({
    "Low risk":    buckets_low,
    "Default risk": buckets_high,
}, index=labels)

st.bar_chart(chart_df, color=["#22c55e", "#ef4444"], height=180)
st.caption(f"🟡 Threshold at {threshold:.0%} — clients above are flagged as default risk")


# ── Client cards ───────────────────────────────────────────────────────────────

st.markdown("<div style='margin-top:1.5rem'></div>", unsafe_allow_html=True)
st.markdown('<p class="section-title">Individual client assessments</p>', unsafe_allow_html=True)

# Filter toggle
filter_col, _ = st.columns([2, 6])
with filter_col:
    show_filter = st.selectbox("Show", ["All clients", "Flagged only", "Low risk only"], label_visibility="collapsed")

show_indices = list(range(len(df)))
if show_filter == "Flagged only":
    show_indices = [i for i in show_indices if preds[i] == 1]
elif show_filter == "Low risk only":
    show_indices = [i for i in show_indices if preds[i] == 0]

# Render in 2 columns
left_col, right_col = st.columns(2)

for idx, i in enumerate(show_indices):
    row   = df.iloc[i]
    prob  = probs[i]
    label, level = risk_label(prob, threshold)
    card_class   = "risk-card-high" if level in ("high",) else "risk-card-low"
    pill_class   = f"pill-{level}"

    sex_str = SEX_MAP.get(int(row.X2), "—")
    edu_str = EDU_MAP.get(int(row.X3), "—")
    mar_str = MAR_MAP.get(int(row.X4), "—")
    credit  = f"NT$ {int(row.X1):,}"
    age     = int(row.X5)
    gpr     = f"{row.good_payment_ratio:.0%}"
    md      = int(row.max_delay)
    trend_icon = "↑" if row.delay_trend > 0 else ("↓" if row.delay_trend < 0 else "→")

    bar_html = prob_bar_html(prob, threshold)

    card_html = f"""
    <div class="risk-card {card_class}">
      <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:8px;">
        <span style="font-size:0.85rem; font-weight:600; color:#e2e8f0;">Client #{i+1:02d}</span>
        <span class="pill {pill_class}">{label}</span>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:4px 16px;
                  font-size:0.75rem; color:#94a3b8; margin-bottom:10px;">
        <span>Age <strong style="color:#cbd5e1;">{age}</strong> · {sex_str}</span>
        <span>Credit <strong style="color:#cbd5e1;">{credit}</strong></span>
        <span>Education: <strong style="color:#cbd5e1;">{edu_str}</strong></span>
        <span>Status: <strong style="color:#cbd5e1;">{mar_str}</strong></span>
        <span>On-time ratio <strong style="color:#cbd5e1;">{gpr}</strong></span>
        <span>Max delay <strong style="color:#cbd5e1;">{md}mo</strong> {trend_icon}</span>
      </div>
      {bar_html}
    </div>
    """

    target_col = left_col if idx % 2 == 0 else right_col
    target_col.markdown(card_html, unsafe_allow_html=True)


# ── Raw payload table (collapsible) ───────────────────────────────────────────

with st.expander("📋  Raw feature table (full payload sent to Lambda)"):
    display_df = df.copy()
    display_df.insert(0, "Client #", [f"#{i+1:02d}" for i in range(len(df))])
    display_df.insert(1, "Default Prob", [f"{p:.3f}" for p in probs])
    display_df.insert(2, "Decision", ["🔴 Default" if p == 1 else "🟢 Low Risk" for p in preds])
    st.dataframe(display_df, use_container_width=True, height=300)
    st.caption(
        f"Payload format: list of {len(df)} records, each with X1–X23 + 4 derived features. "
        "Wrapped in `{{\"body\": \"<json string>\"}}` to match API Gateway proxy format."
    )

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown("""
<div style="display:flex; justify-content:space-between; align-items:center;
            font-size:0.72rem; color:#475569; padding:4px 0;">
  <span>LightGBM · Optuna (30 trials) · ROC AUC 0.809 · Threshold via Precision-Recall F1 max</span>
  <span>UCI Default of Credit Card Clients · 30,000 samples · AWS Lambda + API Gateway</span>
</div>
""", unsafe_allow_html=True)