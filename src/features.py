import pandas as pd, numpy as np
FEATURES = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
TARGET = "non_compliant"
def clean_and_engineer(df: pd.DataFrame):
    df = df.copy()
    keep = [c for c in FEATURES+[TARGET] if c in df.columns]
    df = df[keep]
    for c in FEATURES:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')
    for c in FEATURES:
        if c in df.columns: df[c] = df[c].fillna(df[c].median())
    clips = {"ph":(0,14),"Hardness":(0,1000),"Solids":(0,100000),"Chloramines":(0,20),
             "Sulfate":(0,1000),"Conductivity":(0,2000),"Organic_carbon":(0,50),
             "Trihalomethanes":(0,300),"Turbidity":(0,50)}
    for c,(lo,hi) in clips.items():
        if c in df.columns: df[c] = df[c].clip(lo,hi)
    return df
