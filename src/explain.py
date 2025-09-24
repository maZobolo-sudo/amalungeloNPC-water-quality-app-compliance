import shap, matplotlib.pyplot as plt
def shap_summary(model, X):
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    plt.figure(figsize=(8,5))
    sv = sv[1] if isinstance(sv, list) else sv
    shap.summary_plot(sv, X, show=False)
    return plt.gcf()
