# utils/explain_shap.py
import numpy as np
import shap

def build_shap(model, feature_names, background: np.ndarray = None):
    """
    Build a KernelExplainer backed by real background data (300, 15209).
    Falls back to a single zeros row if background is None.
    """
    feature_names = list(feature_names)

    if background is None:
        background = np.zeros((1, len(feature_names)), dtype=np.float32)

    explainer = shap.KernelExplainer(
        model.predict_proba,
        background,
        link="identity",
    )
    return explainer, feature_names


def explain_shap(explainer, feature_names, vectorizer, processed_text: str, class_names, top_n=10):
    """
    Returns [(word, shap_value), ...] for the predicted class, sorted by abs importance.
    """
    feature_names = list(feature_names)
    x = vectorizer.transform([processed_text]).toarray().astype(np.float32)  # (1, 15209)

    # nsamples controls speed vs accuracy — 100 is a good balance on Streamlit Cloud
    shap_values = explainer.shap_values(x, nsamples=100)
    # shap_values: list of (1, 15209) arrays, one per class

    # Pick class with largest total absolute SHAP contribution
    class_idx = int(np.argmax([np.abs(sv[0]).sum() for sv in shap_values]))

    values = shap_values[class_idx][0]  # shape (15209,)

    pairs = [
        (feat, float(val))
        for feat, val in zip(feature_names, values)
        if val != 0.0
    ]
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:top_n]
