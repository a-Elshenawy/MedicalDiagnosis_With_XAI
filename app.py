import streamlit as st
import numpy as np
import pickle
import os
import gdown
import logging

from utils.preprocess import basic_clean
from utils.translate import is_arabic, to_english, to_arabic
from utils.explain_lime import build_lime, explain_lime
from utils.rules import apply_rules

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Disease Prediction AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# CUSTOM CSS
# =========================
st.markdown(""" 
<style>
/* (UNCHANGED CSS — omitted here for brevity, keep yours exactly as-is) */
</style>
""", unsafe_allow_html=True)

# =========================
# LOGGING
# =========================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

if "init_logged" not in st.session_state:
    logging.info("Session started")
    st.session_state.init_logged = True

# =========================
# MODEL LOADER
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

@st.cache_resource
def load_model(file_id, path):
    os.makedirs(MODELS_DIR, exist_ok=True)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=True)

    with open(path, "rb") as f:
        return pickle.load(f)

MODEL_V1_ID = "1dk4NtpEGTN1kD9emP7WAgSGS28c0LiOF"
MODEL_V2_ID = "1cM_go5CgkA0y45GRSsV5czcxXwa-el_4"

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Model Control")
    version = st.selectbox("Model Version", ["v1", "v2"])

    st.markdown("---")
    st.markdown("### 🧠 Explainer")
    explainer_choice = st.radio("Method", ["LIME"], index=0)

# =========================
# LOAD MODEL
# =========================
if version == "v1":
    data = load_model(MODEL_V1_ID, os.path.join(MODELS_DIR, "model-v1.pkl"))
else:
    data = load_model(MODEL_V2_ID, os.path.join(MODELS_DIR, "model-v2.pkl"))

model         = data["model"]
vectorizer    = data["vectorizer"]
label_map     = data["label_map"]
class_names   = data["class_names"]
feature_names = data["feature_names"]

# =========================
# LIME EXPLAINER
# =========================
@st.cache_resource
def get_lime(_class_names):
    return build_lime(_class_names)

lime_explainer = get_lime(class_names)

# =========================
# HERO
# =========================
st.markdown('<div class="hero-title">🩺 Disease Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Arabic & English · LIME Explainability</div>', unsafe_allow_html=True)

# =========================
# SESSION STATE
# =========================
if "symptom_input" not in st.session_state:
    st.session_state.symptom_input = ""
if "result" not in st.session_state:
    st.session_state.result = None
if "display_arabic" not in st.session_state:
    st.session_state.display_arabic = None

# =========================
# EXAMPLES
# =========================
examples = ["fever headache", "sore throat cough", "ألم في الرأس", "stomach pain vomiting"]
cols = st.columns(len(examples))

for i, ex in enumerate(examples):
    if cols[i].button(ex):
        st.session_state.symptom_input = ex
        st.rerun()

# =========================
# INPUT
# =========================
user_input = st.text_area(
    "Enter symptoms (Arabic or English):",
    value=st.session_state.symptom_input,
    height=110,
)

predict_clicked = st.button("🔍 Predict")

# =========================
# PREDICT
# =========================
if predict_clicked:
    if not user_input.strip():
        st.warning("Please enter symptoms.")
        st.stop()

    arabic_input = is_arabic(user_input)
    processed = to_english(user_input) if arabic_input else user_input
    processed = basic_clean(processed)

    X = vectorizer.transform([processed])
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    conf = float(np.max(probs))

    disease = label_map.get(pred, str(pred))
    rule_msg = apply_rules(conf)

    lime_exp = explain_lime(lime_explainer, model, vectorizer, processed)

    def translate_pairs(pairs):
        return [(to_arabic(w), v) for w, v in pairs]

    st.session_state.result = {
        "arabic_input": arabic_input,
        "disease_en": disease,
        "disease_ar": to_arabic(disease),
        "conf": conf,
        "rule_en": rule_msg,
        "rule_ar": to_arabic(rule_msg),
        "lime_en": lime_exp,
        "lime_ar": translate_pairs(lime_exp),
    }

    st.session_state.display_arabic = arabic_input

# =========================
# RESULTS
# =========================
if st.session_state.result:
    r = st.session_state.result
    ar = st.session_state.display_arabic

    disease_show = r["disease_ar"] if ar else r["disease_en"]
    rule_show = r["rule_ar"] if ar else r["rule_en"]
    lime_show = r["lime_ar"] if ar else r["lime_en"]
    conf = r["conf"]

    st.markdown(f"""
    <div class="card card-accent">
        <div class="disease-badge">{disease_show}</div>
        <p>Confidence: {conf:.2f}</p>
        <p>Rule: {rule_show}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 LIME Explanation")

    for word, val in lime_show:
        st.write(f"{word}: {val:.4f}")
