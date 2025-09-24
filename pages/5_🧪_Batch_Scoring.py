import streamlit as st, pandas as pd, joblib
from pathlib import Path
from src.models import predict_df
st.title("🧪 Batch Scoring")
WORKSPACE = st.secrets.get("workspace_key","default")
model_path = Path(f"tenants/{WORKSPACE}/models/model.joblib")
if not model_path.exists(): st.error("No trained model found. Train on the Model page first."); st.stop()
model = joblib.load(model_path)
uploaded = st.file_uploader("Upload lab results to score (CSV preferred)", type=["csv","parquet"])
if not uploaded: st.info("Upload a dataset to score."); st.stop()
df_new = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_parquet(uploaded)
scored = predict_df(model, df_new); st.write("Scored rows:", len(scored)); st.dataframe(scored.head())
thr = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.01); scored["flag_high_risk"] = (scored["risk_prob"]>=thr).astype(int)
st.write("High-risk count:", int(scored["flag_high_risk"].sum()))
csv = scored.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Scored CSV", data=csv, file_name="scored_lab_results.csv", mime="text/csv")
