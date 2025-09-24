import streamlit as st, pandas as pd, os
from pathlib import Path
st.title("🛠️ Ops & Agent")
WORKSPACE = st.secrets.get("workspace_key","default")
logpath = f"tenants/{WORKSPACE}/reports/agent_log.csv"
message = st.text_area("Message", "Alert: High non-compliance risk at Plant X. Please investigate and retest.")
severity = st.selectbox("Severity", ["info","warning","critical"])
if st.button("Log Alert"):
    import pandas as pd, os
    from datetime import datetime
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    row = pd.DataFrame([{"timestamp":ts,"severity":severity,"message":message}])
    os.makedirs(f"tenants/{WORKSPACE}/reports", exist_ok=True)
    if os.path.exists(logpath): old=pd.read_csv(logpath); pd.concat([old,row]).to_csv(logpath,index=False)
    else: row.to_csv(logpath,index=False)
    st.success(f"Logged to {logpath}")
if os.path.exists(logpath): st.dataframe(pd.read_csv(logpath).tail(50))
else: st.info("No alerts logged yet.")
