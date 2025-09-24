import streamlit as st, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
from src.features import clean_and_engineer, FEATURES, TARGET
st.title("📊 EDA")
WORKSPACE = st.secrets.get("workspace_key","default")
p = Path(f"tenants/{WORKSPACE}/data/intake_clean.csv")
if not p.exists(): st.warning("No tenant dataset found. Use Data Intake Wizard."); st.stop()
df = pd.read_csv(p); st.write("Raw shape:", df.shape); st.dataframe(df.head())
df_clean = clean_and_engineer(df); st.write("After cleaning:", df_clean.shape)
st.subheader("Missingness (original)")
missing = df.isna().mean().sort_values(ascending=False)
fig = plt.figure(figsize=(6,3)); plt.plot(missing.index, missing.values, marker='o'); plt.xticks(rotation=45, ha='right'); plt.ylabel("Missing fraction")
st.pyplot(fig)
st.subheader("Feature distributions (cleaned)")
col1, col2 = st.columns(2)
for i,c in enumerate(FEATURES):
    fig2 = plt.figure(figsize=(4,3)); plt.hist(df_clean[c].values, bins=30); plt.title(c)
    (col1 if i%2==0 else col2).pyplot(fig2)
st.session_state["eda_df_clean"] = df_clean
if TARGET in df.columns: st.write("Target positive rate:", df[TARGET].mean())
