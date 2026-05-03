from lime.lime_text import LimeTextExplainer
def build_lime(class_names):
    return LimeTextExplainer(class_names=class_names)
def explain_lime(explainer, model, vectorizer, text, top_k=6):
    def predict_proba(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)
    exp = explainer.explain_instance(
        text,
        predict_proba,
        num_features=top_k,
        num_samples=300
    )
    return exp.as_list()     
