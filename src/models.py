import os, joblib, numpy as np, pandas as pd, json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support
from .features import FEATURES, TARGET, clean_and_engineer
MODEL_PATH = "tenants/{WORKSPACE}/models/model.joblib"
def train_model(df: pd.DataFrame, random_state=42):
    df = clean_and_engineer(df)
    if TARGET not in df.columns: raise ValueError(f"Need '{TARGET}' to train.")
    X, y = df[FEATURES], df[TARGET].astype(int)
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=random_state,stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=random_state, n_jobs=-1).fit(Xtr,ytr)
    yhat = model.predict(Xte); proba = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba); acc = accuracy_score(yte, yhat)
    prec,rec,f1,_ = precision_recall_fscore_support(yte, yhat, average="binary")
    metrics = {"auc":float(auc),"accuracy":float(acc),"precision":float(prec),"recall":float(rec),"f1":float(f1),
               "n_train":int(len(Xtr)),"n_test":int(len(Xte))}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    # Save training baseline stats
    baseline = {
        "feature_means": {c: float(np.mean(X[c])) for c in FEATURES},
        "feature_stds":  {c: float(np.std(X[c]))  for c in FEATURES},
        "class_pos_rate": float(y.mean()),
    }
    with open(os.path.join(os.path.dirname(MODEL_PATH), "baseline_stats.json"), "w") as f:
        json.dump(baseline, f)
    return model, metrics
def load_model():
    import joblib
    return joblib.load(MODEL_PATH)
def predict_df(model, df: pd.DataFrame):
    from .features import clean_and_engineer, FEATURES
    d = clean_and_engineer(df); proba = model.predict_proba(d[FEATURES])[:,1]
    pred = (proba>=0.5).astype(int); d["risk_prob"]=proba; d["pred_non_compliant"]=pred; return d
