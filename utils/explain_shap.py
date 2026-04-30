import shap
import numpy as np

def build_shap_explainer(model, X_sample):
    return shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")


def explain_shap(explainer, X_vector, feature_names, top_k=10):

    shap_values = explainer.shap_values(X_vector)

    if isinstance(shap_values, list):
        shap_values = shap_values[np.argmax(np.abs(shap_values).mean(axis=1))]

    values = shap_values[0]

    explanation = list(zip(feature_names, values))
    explanation = sorted(explanation, key=lambda x: abs(x[1]), reverse=True)

    return explanation[:top_k]
