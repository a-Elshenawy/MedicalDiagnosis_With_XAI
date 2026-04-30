import streamlit as st
import numpy as np
import pickle
import os
import gdown

from utils.preprocess import basic_clean
from utils.translate import is_arabic, to_english, to_arabic
from utils.explain_lime import build_lime, explain_lime
from utils.explain_shap import build_shap_explainer, explain_shap
from utils.rules import apply_rules

# =========================
# GOOGLE DRIVE MODEL LOADER
# =========================

@st.cache_resource
def load_model_from_drive(file_id, output_path):
    if not os.path.exists(output_path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, output_path, quiet=False)

    with open(output_path, "rb") as f:
        return pickle.load(f)

# =========================
# MODEL IDS (PUT YOUR IDs HERE)
# =========================
MODEL_V1_ID = "1dk4NtpEGTN1kD9emP7WAgSGS28c0LiOF"
MODEL_V2_ID = "1cM_go5CgkA0y45GRSsV5czcxXwa-el_4"

# =========================
# UI - MODEL SELECTION
# =========================
st.sidebar.title("⚙️ Model Control")
version = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])

if version == "v1":
    data = load_model_from_drive(MODEL_V1_ID, "model-v1.pkl")
else:
    data = load_model_from_drive(MODEL_V2_ID, "model-v2.pkl")

model = data["model"]
vectorizer = data["vectorizer"]
label_map = data["label_map"]
class_names = data["class_names"]
feature_names = data["feature_names"]

# =========================
# EXPLAINERS
# =========================
lime_explainer = build_lime(class_names)

X_sample = vectorizer.transform(["fever headache cough"] * 50)
shap_explainer = build_shap_explainer(model, X_sample)

# =========================
# UI HEADER
# =========================
st.title("🩺 Disease Prediction AI System")
st.write("Arabic + English | LIME + SHAP | Model v1/v2")

# =========================
# EXAMPLES
# =========================
examples = [
    "I have fever and headache",
    "sore throat and cough",
    "ألم في الرأس",
    "stomach pain and vomiting"
]

st.subheader("💡 Examples")
cols = st.columns(len(examples))

example_input = None
for i, ex in enumerate(examples):
    if cols[i].button(ex):
        example_input = ex

# =========================
# USER INPUT
# =========================
user_input = st.text_area("Enter symptoms:", value=example_input or "")

# =========================
# PREDICT
# =========================
if st.button("Predict"):

    if not user_input.strip():
        st.warning("Please enter symptoms")
        st.stop()

    arabic = is_arabic(user_input)

    processed = to_english(user_input) if arabic else user_input
    processed = basic_clean(processed)

    X = vectorizer.transform([processed])

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    conf = float(np.max(probs))

    disease = label_map.get(pred, str(pred))
    rule_msg = apply_rules(conf)

    # =========================
    # LIME
    # =========================
    lime_exp = explain_lime(lime_explainer, model, vectorizer, processed)

    # =========================
    # SHAP
    # =========================
    shap_exp = explain_shap(shap_explainer, X, feature_names)

    # =========================
    # SHAPES
    # =========================
    input_shape = len([processed])
    vector_shape = X.shape
    prob_shape = probs.shape

    # =========================
    # ENGLISH OUTPUT
    # =========================
    st.subheader("🇬🇧 English Output")

    st.write("Disease:", disease)
    st.write("Confidence:", conf)
    st.write("Rule:", rule_msg)

    st.subheader("📊 Data Shapes")
    st.write("Input shape:", input_shape)
    st.write("Vector shape:", vector_shape)
    st.write("Probability shape:", prob_shape)

    st.subheader("🧠 LIME Explanation")
    for word, weight in lime_exp:
        st.write(f"{word}: {weight:.4f}")

    st.subheader("📊 SHAP Explanation")
    for word, weight in shap_exp:
        st.write(f"{word}: {weight:.4f}")

    # =========================
    # ARABIC OUTPUT
    # =========================
    st.subheader("🇸🇦 Arabic Output")

    st.write("المرض:", to_arabic(disease) if arabic else disease)
    st.write("الثقة:", conf)
    st.write("القاعدة:", to_arabic(rule_msg) if arabic else rule_msg)

    st.subheader("🧠 التفسير (LIME)")
    for word, weight in lime_exp:
        w = to_arabic(word) if arabic else word
        st.write(f"{w}: {weight:.4f}")

    st.subheader("📊 التفسير (SHAP)")
    for word, weight in shap_exp:
        w = to_arabic(word) if arabic else word
        st.write(f"{w}: {weight:.4f}")
