
from lime.lime_text import LimeTextExplainer

def build_lime(class_names, vectorizer):
    tokenizer = vectorizer.build_tokenizer()
    # LIME's split_expression can be a callable in newer versions,
    # but the cleanest approach is to pre-tokenize
    return LimeTextExplainer(
        class_names=class_names,
        bow=True
    )

def explain_lime(explainer, model, vectorizer, text, top_k=6):
    analyzer = vectorizer.build_analyzer()   # respects n-gram range + stop words

    def predict_proba(texts):
        X = vectorizer.transform(texts)
        return model.predict_proba(X)

    # Feed LIME the pre-analyzed n-gram string instead of raw text
    ngram_text = " ".join(analyzer(text))    # e.g. "fever headache fever headache chest pain"

    exp = explainer.explain_instance(
        ngram_text,
        predict_proba,
        num_features=top_k,
        num_samples=500
    )
    return exp.as_list()
