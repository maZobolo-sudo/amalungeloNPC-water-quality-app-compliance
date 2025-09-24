import streamlit as st, pandas as pd
from pathlib import Path
from src.features import FEATURES, TARGET
st.title("📥 Data Intake Wizard")
uploaded = st.file_uploader("Upload lab results CSV", type=["csv"])
if not uploaded:
    st.info("Upload a CSV or download a template below.")
    templ = pd.DataFrame(columns=FEATURES+[TARGET])
    st.download_button("Download Template CSV", templ.to_csv(index=False).encode("utf-8"),
                       file_name="water_quality_template.csv", mime="text/csv")
    st.stop()
df_raw = pd.read_csv(uploaded); st.write("Raw shape:", df_raw.shape)
st.subheader("Column Mapping")
mapping = {}
for need in FEATURES:
    options = ["-- select --"] + list(df_raw.columns)
    mapping[need] = st.selectbox(f"Map to '{need}'", options, index=options.index(need) if need in df_raw.columns else 0)
if any(v=="-- select --" for v in mapping.values()):
    st.warning("Please map all required columns."); st.stop()
df_mapped = df_raw.rename(columns={v:k for k,v in mapping.items() if v!="-- select --"})
st.subheader("Units")
u_conduct = st.selectbox("Conductivity", ["µS/cm","mS/cm"], index=0)
u_thm = st.selectbox("Trihalomethanes", ["µg/L","mg/L"], index=0)
if u_conduct=="mS/cm" and "Conductivity" in df_mapped.columns: df_mapped["Conductivity"]*=1000.0
if u_thm=="mg/L" and "Trihalomethanes" in df_mapped.columns: df_mapped["Trihalomethanes"]*=1000.0
WORKSPACE = st.secrets.get("workspace_key","default")
out = Path(f"tenants/{WORKSPACE}/data/intake_clean.csv"); out.parent.mkdir(parents=True, exist_ok=True)
df_mapped.to_csv(out, index=False); st.success(f"Saved to {out}")
st.dataframe(df_mapped.head())
