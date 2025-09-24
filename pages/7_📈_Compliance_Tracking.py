import streamlit as st, pandas as pd, matplotlib.pyplot as plt, datetime
from pathlib import Path
from src.features import FEATURES
from src.monitor import load_baseline, feature_drift, append_history, load_history
st.title("📈 Compliance & Trends")
WORKSPACE = st.secrets.get("workspace_key","default")
REPORT_DIR = Path(f"tenants/{WORKSPACE}/reports"); REPORT_DIR.mkdir(parents=True, exist_ok=True)
st.markdown("Upload a **scored CSV** (from Batch Scoring) to compute compliance KPIs and monitor drift vs training.")
file = st.file_uploader("Upload scored CSV (requires 'risk_prob' column)", type=["csv"])
date_str = st.text_input("Monitoring date (YYYY-MM-DD)", value=str(datetime.date.today()))
thr = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.01)
if file:
    df = pd.read_csv(file)
    if "risk_prob" not in df.columns:
        st.error("No 'risk_prob' column found. Please upload the scored CSV from the Batch Scoring page.")
        st.stop()
    df["flag_high_risk"] = (df["risk_prob"] >= thr).astype(int)
    hi = int(df["flag_high_risk"].sum()); total = len(df); rate = df["flag_high_risk"].mean()
    c1, c2, c3 = st.columns(3)
    c1.metric("High-risk count", hi); c2.metric("High-risk rate", f"{rate:.3f}"); c3.metric("Total scored", total)
    baseline = load_baseline(WORKSPACE)
    if baseline:
        st.subheader("Drift (KS test vs baseline)")
        drift = feature_drift(df, baseline, FEATURES); st.dataframe(drift.head(10))
    else:
        st.info("Train a model first to enable drift checks.")
    snapshot = {"date": date_str, "threshold": thr, "high_risk_rate": float(rate), "high_risk_count": hi, "total": total}
    if st.button("📌 Record snapshot"): 
        p = append_history(WORKSPACE, snapshot); st.success(f"Snapshot saved to {p}")
st.subheader("Trend: High-risk rate over time")
hist = load_history(WORKSPACE)
if hist.empty:
    st.info("No history yet. Record a snapshot above.")
else:
    hist["date"] = pd.to_datetime(hist["date"], errors="coerce"); hist = hist.sort_values("date")
    fig = plt.figure(figsize=(7,3)); plt.plot(hist["date"], hist["high_risk_rate"], marker="o")
    plt.ylabel("High-risk rate"); plt.xlabel("Date"); st.pyplot(fig); st.dataframe(hist.tail(20))
