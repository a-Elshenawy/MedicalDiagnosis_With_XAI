import streamlit as st
import numpy as np
import pickle
import os
import gdown
import logging

from utils.preprocess import basic_clean
from utils.translate import is_arabic, to_english, to_arabic
from utils.explain_lime import build_lime, explain_lime
from utils.explain_shap import build_shap_explainer, explain_shap
from utils.rules import apply_rules

# =========================
# LOGGING SETUP
# =========================
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    filename="logs/app.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# =========================
# SESSION INITIAL LOG CONTROL
# =========================
if "init_logged" not in st.session_state:
    logging.info("App session started")
    st.session_state.init_logged = True

# =========================
# MODEL LOADER
# =========================
@st.cache_resource
def load_model(file_id, path):
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

    with open(path, "rb") as f:
        return pickle.load(f)

# =========================
# MODEL IDS
# =========================
MODEL_V1_ID = "1dk4NtpEGTN1kD9emP7WAgSGS28c0LiOF"
MODEL_V2_ID = "1cM_go5CgkA0y45GRSsV5czcxXwa-el_4"

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Model Control")
version = st.sidebar.selectbox("Choose Model Version", ["v1", "v2"])

# log ONLY when user changes model
if "last_version" not in st.session_state:
    st.session_state.last_version = version

if version != st.session_state.last_version:
    logging.info(f"Model switched to: {version}")
    st.session_state.last_version = version

# =========================
# LOAD MODEL
# =========================
if version == "v1":
    data = load_model(MODEL_V1_ID, "model-v1.pkl")
else:
    data = load_model(MODEL_V2_ID, "model-v2.pkl")

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
# UI
# =========================
st.title("🩺 Disease Prediction AI System")
st.write("Arabic + English | LIME + SHAP | Clean Logging System")

# =========================
# SESSION INPUT STATE
# =========================
if "symptom_input" not in st.session_state:
    st.session_state.symptom_input = ""

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

for i, ex in enumerate(examples):
    if cols[i].button(ex):
        st.session_state.symptom_input = ex
        logging.info(f"Example used: {ex}")

# =========================
# INPUT
# =========================
user_input = st.text_area("Enter symptoms:", key="symptom_input")

# =========================
# PREDICT
# =========================
if st.button("Predict"):

    if not user_input or not user_input.strip():
        st.warning("Please enter symptoms")
        logging.warning("Empty input submitted")
        st.stop()

    logging.info(f"User input: {user_input}")

    arabic = is_arabic(user_input)

    processed = to_english(user_input) if arabic else user_input
    processed = basic_clean(processed)

    X = vectorizer.transform([processed])

    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    conf = float(np.max(probs))

    disease = label_map.get(pred, str(pred))
    rule_msg = apply_rules(conf)

    logging.info(f"Prediction: {disease} | Confidence: {conf:.2f}")

    # =========================
    # EXPLANATIONS
    # =========================
    lime_exp = explain_lime(lime_explainer, model, vectorizer, processed)
    shap_exp = explain_shap(shap_explainer, X, feature_names)

    # =========================
    # SHAPES
    # =========================
    st.subheader("📊 Shapes")
    st.write("Input shape:", len([processed]))
    st.write("Vector shape:", X.shape)
    st.write("Probability shape:", probs.shape)

    # =========================
    # OUTPUT ENGLISH
    # =========================
    st.subheader("🇬🇧 English Output")
    st.write("Disease:", disease)
    st.write("Confidence:", conf)
    st.write("Rule:", rule_msg)

    st.subheader("🧠 LIME")
    for w, v in lime_exp:
        st.write(f"{w}: {v:.4f}")

    st.subheader("📊 SHAP")
    for w, v in shap_exp:
        st.write(f"{w}: {v:.4f}")

    # =========================
    # OUTPUT ARABIC
    # =========================
    st.subheader("🇸🇦 Arabic Output")
    st.write("المرض:", to_arabic(disease) if arabic else disease)
    st.write("الثقة:", conf)
    st.write("القاعدة:", to_arabic(rule_msg) if arabic else rule_msg)

# =========================
# LOG VIEWER (FIXED TOGGLE)
# =========================
if "show_logs" not in st.session_state:
    st.session_state.show_logs = False

if st.sidebar.button("Toggle Logs"):
    st.session_state.show_logs = not st.session_state.show_logs

if st.session_state.show_logs:
    with open("logs/app.log", "r") as f:
        log_data = f.read()

    st.sidebar.text_area("Logs", log_data, height=300)

    st.sidebar.download_button(
        "Download Logs",
        data=log_data,
        file_name="app.log"
    )
