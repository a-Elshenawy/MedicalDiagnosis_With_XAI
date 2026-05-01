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
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

/* ── Global ── */
html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

/* ── Background ── */
.stApp {
  background: linear-gradient(135deg, #0d1117 0%, #0f1923 50%, #0d1117 100%);
  color: #e6edf3;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] .stButton button { color: #8b949e !important; }

/* ── Cards ── */
.card { background: #161b22; border: 1px solid #21262d; border-radius: 12px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem; }
.card-accent  { border-left: 4px solid #58a6ff; }
.card-green   { border-left: 4px solid #3fb950; }
.card-purple  { border-left: 4px solid #bc8cff; }
.card-orange  { border-left: 4px solid #f0883e; }

/* ── Title ── */
.hero-title { font-size: 2.4rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
.hero-sub   { color: #8b949e; font-size: 0.95rem; margin-bottom: 2rem; font-weight: 300; }

/* ── Disease badge ── */
.disease-badge {
  display: inline-block;
  background: linear-gradient(90deg, #1f6feb, #388bfd);
  color: #fff; font-size: 1.4rem; font-weight: 700;
  padding: 0.5rem 1.2rem; border-radius: 8px;
  letter-spacing: 0.01em; margin-bottom: 0.6rem;
}

/* ── Confidence bar ── */
.conf-label { font-size: 0.85rem; color: #8b949e; margin-bottom: 2px; }

/* ── Explain rows ── */
.explain-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 0.35rem 0; border-bottom: 1px solid #21262d;
  font-family: 'IBM Plex Mono', monospace; font-size: 0.88rem;
}
.explain-row:last-child { border-bottom: none; }
.word-label { color: #e6edf3; }
.val-pos { color: #3fb950; font-weight: 600; }
.val-neg { color: #f85149; font-weight: 600; }

/* ── Example buttons ── */
.stButton > button {
  background: #21262d !important; color: #c9d1d9 !important;
  border: 1px solid #30363d !important; border-radius: 8px !important;
  font-family: 'Sora', sans-serif !important; font-size: 0.82rem !important;
  padding: 0.4rem 0.8rem !important; transition: all 0.2s !important;
}
.stButton > button:hover { background: #1f6feb !important; border-color: #388bfd !important; color: #fff !important; }

/* ── Predict button ── */
.predict-btn > button {
  background: linear-gradient(90deg, #1f6feb, #388bfd) !important;
  color: #fff !important; font-size: 1rem !important; font-weight: 600 !important;
  border: none !important; border-radius: 10px !important;
  padding: 0.65rem 2.5rem !important; letter-spacing: 0.02em !important;
}
.predict-btn > button:hover { background: linear-gradient(90deg, #388bfd, #58a6ff) !important; }

/* ── Translate button ── */
.translate-btn > button {
  background: #21262d !important; border: 1px solid #388bfd !important;
  color: #58a6ff !important; font-size: 0.88rem !important; border-radius: 8px !important;
}
.translate-btn > button:hover { background: #1f6feb22 !important; }

/* ── Text area ── */
.stTextArea textarea {
  background: #0d1117 !important; border: 1px solid #30363d !important;
  border-radius: 10px !important; color: #e6edf3 !important;
  font-family: 'Sora', sans-serif !important; font-size: 1rem !important;
}
.stTextArea textarea:focus { border-color: #388bfd !important; box-shadow: 0 0 0 3px #1f6feb33 !important; }

/* ── RTL support ── */
.rtl-text { direction: rtl; text-align: right; font-size: 1.05rem; }

/* ── Section headers ── */
.section-header {
  font-size: 0.78rem; font-weight: 600; letter-spacing: 0.1em;
  text-transform: uppercase; color: #8b949e; margin-bottom: 0.8rem;
}

/* ── Divider ── */
hr.subtle { border: none; border-top: 1px solid #21262d; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# =========================
# LOGGING SETUP
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
MODEL_V3_ID = "1puo1OpWE8dWxQwOKPUUDxkVz4zpCrlBr"   # ← replace when ready

# Absolute base so all file ops work identically on Streamlit Cloud and locally
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.markdown("### ⚙️ Model Control")
    version = st.selectbox("Model Version", ["v1", "v2", "v3"])
    if "last_version" not in st.session_state:
        st.session_state.last_version = version
    if version != st.session_state.last_version:
        logging.info(f"Switched model to {version}")
        st.session_state.last_version = version

    st.markdown("---")
    if "show_logs" not in st.session_state:
        st.session_state.show_logs = False
    if st.button("📋 Toggle Logs"):
        st.session_state.show_logs = not st.session_state.show_logs
    if st.session_state.show_logs:
        try:
            with open("logs/app.log", "r") as f:
                log_data = f.read()
            st.text_area("Logs", log_data, height=260)
            st.download_button("⬇️ Download Logs", log_data, file_name="app.log")
        except FileNotFoundError:
            st.info("No logs yet.")

# =========================
# LOAD MODEL
# =========================
if version == "v1":
    data = load_model(MODEL_V1_ID, os.path.join(MODELS_DIR, "model-v1.pkl"))
elif version == "v2":
    data = load_model(MODEL_V2_ID, os.path.join(MODELS_DIR, "model-v2.pkl"))
else:
    data = load_model(MODEL_V3_ID, os.path.join(MODELS_DIR, "model-v3.pkl"))

model         = data["model"]
vectorizer    = data["vectorizer"]
label_map     = data["label_map"]
class_names   = data["class_names"]
feature_names = data["feature_names"]

# ✅ normalize keys to Python int
label_map = {int(k): v for k, v in label_map.items()}

st.write("Has predict_proba:", hasattr(model, "predict_proba"))

# =========================
# EXPLAINERS
# =========================
@st.cache_resource
def get_lime(_class_names):
    return build_lime(_class_names)

lime_explainer = get_lime(class_names)

# =========================
# HERO
# =========================
st.markdown('<div class="hero-title">🩺 Disease Prediction AI</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Arabic & English · LIME Explainability · Bilingual Output</div>', unsafe_allow_html=True)

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
st.markdown('<div class="section-header">💡 Quick Examples</div>', unsafe_allow_html=True)
examples = ["fever headache", "sore throat cough", "ألم في الرأس", "stomach pain vomiting"]
cols = st.columns(len(examples))
for i, ex in enumerate(examples):
    if cols[i].button(ex, key=f"ex_{i}"):
        st.session_state.symptom_input = ex
        logging.info(f"Example used: {ex}")
        st.rerun()

# =========================
# INPUT
# =========================
user_input = st.text_area(
    "Enter symptoms (Arabic or English):",
    value=st.session_state.symptom_input,
    placeholder="e.g. fever, headache, cough | حمى، صداع، سعال",
    height=110,
    key="symptom_input"
)

col_pred, _ = st.columns([1, 4])
with col_pred:
    st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
    predict_clicked = st.button("🔍 Predict", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICT
# =========================
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one symptom.")
        logging.warning("Empty input submitted")
        st.stop()

    logging.info(f"Input: {user_input}")
    arabic_input = is_arabic(user_input)
    processed    = to_english(user_input) if arabic_input else user_input
    processed    = basic_clean(processed)

    X     = vectorizer.transform([processed])
    #pred  = model.predict(X)[0]
    pred = int(np.array(model.predict(X)).flatten()[0])
    probs = model.predict_proba(X)[0]
    conf  = float(np.max(probs))

    disease  = label_map.get(pred, str(pred))
    rule_msg = apply_rules(conf)
    logging.info(f"Prediction: {disease} | Confidence: {conf:.2f}")

    # Run LIME explainer
    lime_exp = explain_lime(lime_explainer, model, vectorizer, processed)

    # Translate explanations to Arabic if needed
    def translate_pairs(pairs):
        return [(to_arabic(w), v) for w, v in pairs]

    st.session_state.result = {
        "arabic_input": arabic_input,
        "disease_en":   disease,
        "disease_ar":   to_arabic(disease),
        "conf":         conf,
        "rule_en":      rule_msg,
        "rule_ar":      to_arabic(rule_msg),
        "lime_en":      lime_exp,
        "lime_ar":      translate_pairs(lime_exp),
    }
    st.session_state.display_arabic = arabic_input
    st.write({
        "pred_clean": pred,
        "pred_type_clean": str(type(pred)),
        "label_map_key_type_after_fix": str(type(list(label_map.keys())[0])),
        "disease_resolved": disease,
    })
    st.write({
        "pred": pred,
        "exists_in_map": pred in label_map,
        "max_key": max(label_map.keys()),
        "min_key": min(label_map.keys()),
        "num_keys": len(label_map)
    })
# =========================
# RESULTS
# =========================
if st.session_state.result:
    r  = st.session_state.result
    ar = st.session_state.display_arabic

    # ── Translate toggle ──
    st.markdown("<hr class='subtle'>", unsafe_allow_html=True)
    col_toggle, _ = st.columns([1, 4])
    with col_toggle:
        toggle_label = "🌐 عرض بالعربية" if not ar else "🌐 Show in English"
        st.markdown('<div class="translate-btn">', unsafe_allow_html=True)
        if st.button(toggle_label, key="lang_toggle"):
            st.session_state.display_arabic = not ar
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Resolve display values ──
    disease_show = r["disease_ar"] if ar else r["disease_en"]
    rule_show    = r["rule_ar"]    if ar else r["rule_en"]
    lime_show    = r["lime_ar"]    if ar else r["lime_en"]
    conf         = r["conf"]
    rtl          = 'class="rtl-text"' if ar else ""

    # ── Main result card ──
    flag       = "🇸🇦" if ar else "🇬🇧"
    lang_label = "Arabic" if ar else "English"
    st.markdown(f"""
    <div class="card card-accent">
        <div class="section-header">{flag} Diagnosis — {lang_label}</div>
        <div class="disease-badge" {rtl}>{disease_show}</div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(conf, text=f"{conf*100:.1f}%  ({rule_show})")

    # ── LIME Explanation ──
    lime_title = "🧠 شرح LIME" if ar else "🧠 LIME Explanation"
    st.markdown(f"**{lime_title}**")
    st.markdown("---")

    if lime_show:
        for word, val in lime_show:
            val_color = "#3fb950" if val >= 0 else "#f85149"
            sign      = "+" if val >= 0 else ""
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"<span style='color:#e6edf3;font-family:monospace'>{word}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span style='color:{val_color};font-weight:600;font-family:monospace'>{sign}{val:.4f}</span>", unsafe_allow_html=True)
    else:
        st.markdown(f"<span style='color:#8b949e;font-size:0.85rem'>{'لا توجد بيانات' if ar else 'No data'}</span>", unsafe_allow_html=True)
