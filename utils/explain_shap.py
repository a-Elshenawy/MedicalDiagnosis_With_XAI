"""
utils/explain_shap.py
SHAP-based explainability for the disease prediction model.

Key design decisions:
- feature_names comes directly from the pickle (data["feature_names"]),
  NOT from vectorizer.get_feature_names_out(), so the array size is
  always guaranteed to match what the model was trained on.
- Background is a single zero-vector sized to match feature_names length.
- predict_proba receives already-vectorized dense arrays from SHAP, so we
  pass them straight to model.predict_proba (no re-vectorization).
"""

import numpy as np
import shap


def build_shap(model, feature_names):
    """
    Build a SHAP KernelExplainer.

    Args:
        model        : trained sklearn classifier (from pickle)
        feature_names: list/array of feature names (data["feature_names"])

    Returns:
        (explainer, feature_names_array)
    """
    feature_names = np.array(feature_names)
    n_features = len(feature_names)

    # One zero-row background — SHAP perturbs around this baseline.
    # Zero = "word absent", which is the natural baseline for BoW/TF-IDF.
    background = np.zeros((1, n_features))

    explainer = shap.KernelExplainer(model.predict_proba, background)
    return explainer, feature_names


def explain_shap(explainer, feature_names, vectorizer, processed_text,
                 class_names, top_n=10):
    """
    Run SHAP on a single preprocessed (English) text string.

    Args:
        explainer      : KernelExplainer from build_shap()
        feature_names  : np.array from build_shap() — same length as model input
        vectorizer     : fitted vectorizer (used only to transform the text)
        processed_text : cleaned English text
        class_names    : list of class name strings
        top_n          : how many top features to return

    Returns:
        List of (word, shap_value) tuples sorted by |shap_value| descending.
        Only tokens that actually appear in the input are returned.
    """
    # Transform text → dense vector (shape: 1 × n_features)
    X_dense = vectorizer.transform([processed_text]).toarray()

    # Guard: if vectorizer output width differs from feature_names, pad/trim
    n_feat = len(feature_names)
    if X_dense.shape[1] != n_feat:
        padded = np.zeros((1, n_feat))
        cols = min(X_dense.shape[1], n_feat)
        padded[0, :cols] = X_dense[0, :cols]
        X_dense = padded

    # Run SHAP — returns list of arrays (one per class), each shape (1, n_feat)
    shap_values = explainer.shap_values(X_dense, nsamples=100, silent=True)

    # Handle both list-of-arrays (multi-class) and single array (binary)
    if isinstance(shap_values, list):
        # Pick the predicted class (highest absolute contribution)
        pred_class_idx = int(np.argmax([np.abs(sv).sum() for sv in shap_values]))
        values = shap_values[pred_class_idx][0]
    else:
        values = shap_values[0]

    # Only report tokens present in this input
    present_mask = X_dense[0] > 0
    paired = [
        (str(feature_names[i]), float(values[i]))
        for i in range(n_feat)
        if present_mask[i]
    ]

    paired.sort(key=lambda x: abs(x[1]), reverse=True)
    return paired[:top_n]
