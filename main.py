import streamlit as st
import pickle
import numpy as np
from pathlib import Path

# ------------------------------
# App config
# ------------------------------
st.set_page_config(page_title="Fake News Detection", layout="centered")

# ------------------------------
# Model loading (cached)
# ------------------------------
MODELS = {
    "Logistic Regression": "logistic_regression_model.pkl",
    "Linear SVM": "linear_svm_model.pkl",
    "Multinomial Naive Bayes": "multinomial_naive_bayes_model.pkl",
}

@st.cache_resource(show_spinner=False)
def load_model(model_name: str):
    model_path = Path(MODELS[model_name])
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path.resolve()}")
    with open(model_path, "rb") as f:
        return pickle.load(f)

def compute_confidence(model, text: str):
    """Return confidence in %, or None if not available."""
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([text])[0]
            return float(np.max(proba) * 100.0)
        elif hasattr(model, "decision_function"):
            # Pseudo-confidence from margin for SVM
            margin = float(np.ravel(model.decision_function([text]))[0])
            return float(100.0 / (1.0 + np.exp(-abs(margin))))
    except Exception:
        pass
    return None

# ------------------------------
# Sidebar
# ------------------------------
st.sidebar.header("ML Model")
model_choice = st.sidebar.radio("Choose a classifier:", list(MODELS.keys()))
model = load_model(model_choice)

# ------------------------------
# Header
# ------------------------------
st.markdown(
    """
    <div style="text-align:center;">
      <h1 style="margin-bottom:4px;">Fake News Detection</h1>
      <p style="color:#553; margin-top:0;">Classify a news headline or article as <b>REAL</b> or <b>FAKE</b>.</p>
    </div>
    <hr style="margin: 0 0 16px 0;">
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Input
# ------------------------------
user_input = st.text_area("Paste a news headline or article:", height=180, placeholder="Type or paste text here...")

# ------------------------------
# Predict
# ------------------------------
if st.button("Run Prediction"):
    text = user_input.strip()
    if not text:
        st.warning("Please enter some text before prediction.")
    else:
        pred = int(model.predict([text])[0])  # pipeline handles vectorization
        label = "REAL" if pred == 1 else "FAKE"
        confidence = compute_confidence(model, text)

        # Decide colors based on label
        bg_color   = "#eafaf1" if label == "REAL" else "#fdecea"   # light green / light red
        title_col  = "#1D8348" if label == "REAL" else "#C0392B"   # green / red
        conf_color = "#117A65" if label == "REAL" else "#922B21"   # darker green / red

        conf_html = (
            f"<p style='font-size:18px; color:{conf_color}; margin:6px 0 0;'>Confidence: {confidence:.2f}%</p>"
            if confidence is not None else ""
        )

        html = f"""
        <div style="
            padding:20px; border-radius:12px; background-color:{bg_color};
            text-align:center; box-shadow:0 4px 10px rgba(0,0,0,0.08);">
            <h2 style="color:{title_col}; margin:0;">Prediction: {label}</h2>
            {conf_html}
            <p style="color:#666; font-size:13px; margin:10px 0 0;">Model used: {model_choice}</p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

# ------------------------------
# Small tip
# ------------------------------
with st.expander("Why confidence may be missing or low"):
    st.write(
        "- Some models (like Linear SVM) don't provide true probabilities. We compute a pseudo-confidence from the decision margin.\n"
        "- Very short or out-of-domain text can reduce confidence.\n"
        "- Results depend on how the models were trained and preprocessed."
    )
