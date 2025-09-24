import numpy as np, pandas as pd
def generate_water_quality(n=1200, seed=7):
    rng = np.random.default_rng(seed)
    ph = rng.normal(7.2, 0.8, n).clip(4.5, 9.5)
    Hardness = rng.normal(200, 60, n).clip(50, 400)
    Solids = rng.normal(15000, 7000, n).clip(2000, 50000)
    Chloramines = rng.normal(7.2, 1.8, n).clip(1.0, 12.0)
    Sulfate = rng.normal(330, 60, n).clip(100, 500)
    Conductivity = rng.normal(450, 120, n).clip(100, 800)
    Organic_carbon = rng.normal(14, 4, n).clip(2, 30)
    Trihalomethanes = rng.normal(60, 20, n).clip(5, 150)
    Turbidity = rng.normal(3.0, 1.2, n).clip(0.1, 8.0)
    X = pd.DataFrame({
        "ph": ph,"Hardness": Hardness,"Solids": Solids,"Chloramines": Chloramines,
        "Sulfate": Sulfate,"Conductivity": Conductivity,"Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,"Turbidity": Turbidity,
    })
    score = ((np.abs(ph-7.0)>1.5).astype(float)*0.7
             + (Turbidity>5).astype(float)*0.6
             + (Trihalomethanes>80).astype(float)*0.5
             + (Organic_carbon>20).astype(float)*0.4
             + (Chloramines<2.0).astype(float)*0.3
             + (Sulfate>450).astype(float)*0.2
             + (Conductivity>650).astype(float)*0.2
             + (Solids>30000).astype(float)*0.2)
    prob = 1/(1+np.exp(-(score + rng.normal(0,0.3,n) - 1.2)))
    y = (rng.uniform(0,1,n) < prob).astype(int)
    df = X.copy(); df["non_compliant"]=y
    for col in ["Sulfate","Trihalomethanes","ph"]:
        mask = rng.uniform(0,1,n) < 0.07
        df.loc[mask, col] = np.nan
    return df
