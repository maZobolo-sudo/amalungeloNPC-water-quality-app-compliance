import streamlit as st, pandas as pd
from pathlib import Path
from src.features import FEATURES, TARGET
from src.models import train_model
from src.explain import shap_summary
from src.reporting import make_pdf_report, make_pptx_deck
st.title("🤖 Model Training & Explainability")
WORKSPACE = st.secrets.get("workspace_key","default")
p = Path(f"tenants/{WORKSPACE}/data/intake_clean.csv")
if not p.exists(): st.warning("No tenant dataset found. Use Data Intake."); st.stop()
df = pd.read_csv(p)
if TARGET not in df.columns: st.error(f"Dataset must include '{TARGET}' to train."); st.stop()
if st.button("Train / Retrain Model"):
    model, metrics = train_model(df); st.success("Model trained."); st.json(metrics)
    try:
        fig = shap_summary(model, df[FEATURES].sample(min(500, len(df)), random_state=42)); st.pyplot(fig)
    except Exception as e:
        st.info(f"SHAP summary skipped: {e}")
    pdf = make_pdf_report(metrics, f"tenants/{WORKSPACE}/reports/water_quality_report.pdf")
    with open(pdf,"rb") as f: st.download_button("⬇️ Download PDF Report", f, file_name="water_quality_report.pdf")
    ppt = make_pptx_deck(metrics, f"tenants/{WORKSPACE}/reports/water_quality_slides.pptx")
    with open(ppt,"rb") as f: st.download_button("⬇️ Download Slides (PPTX)", f, file_name="water_quality_slides.pptx")
else:
    st.info("Click **Train / Retrain Model** to build the classifier.")
