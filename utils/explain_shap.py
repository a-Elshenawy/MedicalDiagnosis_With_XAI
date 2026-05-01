# utils/explain_shap.py

import numpy as np

def build_shap(model, feature_names):
    """
    Build SHAP-compatible structure.
    For linear models (LogisticRegression), we DON'T need shap library.
    We use coefficients directly (fast + stable).
    """

    # Convert to numpy for safe indexing
    coef = model.coef_              # shape: (n_classes, n_features)
    intercept = model.intercept_    # shape: (n_classes,)

    return {
        "coef": coef,
        "intercept": intercept,
        "feature_names": list(feature_names)
    }, list(feature_names)


def explain_shap(shap_data, feature_names, vectorizer, text, class_names, top_k=8):
    """
    Generate SHAP-like explanation using linear model weights.

    Returns:
        list of (word, importance)
    """

    coef = shap_data["coef"]

    # Transform input text
    X = vectorizer.transform([text])   # shape: (1, n_features)
    x_dense = X.toarray()[0]           # convert to dense vector

    # Get predicted class
    # (we recompute here to stay independent)
    class_idx = np.argmax(np.dot(coef, x_dense))

    # Get weights for that class
    class_weights = coef[class_idx]

    # Element-wise contribution
    contributions = x_dense * class_weights

    # Get indices of non-zero features only (important!)
    non_zero_idx = np.where(x_dense > 0)[0]

    # Collect word contributions
    words = []
    for idx in non_zero_idx:
        word = feature_names[idx]
        weight = contributions[idx]
        words.append((word, weight))

    # Sort by absolute importance
    words = sorted(words, key=lambda x: abs(x[1]), reverse=True)

    return words[:top_k]
