"""
utils/explain_shap.py
SHAP-based explainability for the disease prediction model.
Works with sklearn Pipeline or (vectorizer + model) combos.
"""
 
import numpy as np
import shap
 
 
def build_shap(model, vectorizer, class_names):
    """
    Build a SHAP KernelExplainer using a small background sample
    of zero-vectors (sparse-safe).
 
    Returns the explainer object (cache with @st.cache_resource).
    """
    # background: a single all-zero dense vector — lightweight baseline
    background = np.zeros((1, len(vectorizer.get_feature_names_out())))
 
    def predict_proba(X_dense):
        # SHAP passes dense numpy arrays; vectorizer already applied
        return model.predict_proba(X_dense)
 
    explainer = shap.KernelExplainer(predict_proba, background)
    return explainer, vectorizer.get_feature_names_out()
 
 
def explain_shap(explainer, feature_names, vectorizer, processed_text,
                 class_names, top_n=10):
    """
    Run SHAP on a single preprocessed (English) text input.
 
    Returns:
        List of (word, shap_value) tuples for the top predicted class,
        sorted by absolute importance descending.
    """
    X_dense = vectorizer.transform([processed_text]).toarray()
    shap_values = explainer.shap_values(X_dense, nsamples=100, silent=True)
 
    # shap_values: list of arrays, one per class → shape (1, n_features)
    # Pick the class with the highest mean predicted proba for this input
    # (matches what the model would predict)
    pred_class_idx = int(np.argmax(
        [np.abs(sv).mean() for sv in shap_values]
    ))
 
    values = shap_values[pred_class_idx][0]  # shape: (n_features,)
 
    # Pair feature names with their SHAP value, keep only non-zero tokens
    # that actually appeared in the input
    present = X_dense[0] > 0
    paired = [
        (feature_names[i], float(values[i]))
        for i in range(len(feature_names))
        if present[i]
    ]
 
    # Sort by absolute value
    paired.sort(key=lambda x: abs(x[1]), reverse=True)
    return paired[:top_n]
