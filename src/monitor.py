import os, json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ks_2samp
def load_baseline(workspace: str):
    p = Path(f"tenants/{workspace}/models/baseline_stats.json")
    if not p.exists(): return None
    return json.loads(p.read_text())
def feature_drift(df_new: pd.DataFrame, baseline: dict, features: list):
    rows = []
    for c in features:
        if c not in df_new.columns or c not in baseline.get("feature_means", {}):
            continue
        x = pd.to_numeric(df_new[c], errors="coerce").dropna().values
        ref_mu = baseline["feature_means"][c]
        ref_sigma = baseline["feature_stds"][c] or 1.0
        ref = np.random.normal(ref_mu, ref_sigma, size=min(5000, max(1000, len(x))))
        ks = ks_2samp(x, ref)
        rows.append({"feature": c, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)})
    return pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
def append_history(workspace: str, row: dict, fname: str = "compliance_history.csv"):
    hist_dir = Path(f"tenants/{workspace}/reports"); hist_dir.mkdir(parents=True, exist_ok=True)
    p = hist_dir / fname
    df = pd.DataFrame([row])
    if p.exists():
        old = pd.read_csv(p)
        out = pd.concat([old, df], ignore_index=True)
    else:
        out = df
    out.to_csv(p, index=False)
    return p
def load_history(workspace: str, fname: str = "compliance_history.csv"):
    p = Path(f"tenants/{workspace}/reports") / fname
    if not p.exists(): 
        return pd.DataFrame()
    return pd.read_csv(p)
