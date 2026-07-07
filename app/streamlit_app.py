import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
from PIL import Image

from src.inference.predictor import UAIREPredictor

# -----------------------------------------------------
# Page Config
# -----------------------------------------------------

st.set_page_config(

    page_title="UAIRE",

    page_icon="🛡️",

    layout="wide"

)

# -----------------------------------------------------
# Title
# -----------------------------------------------------

st.title("🛡️ UAIRE")

st.subheader("Universal AI Reliability Engine")

st.markdown(
"""
Predict **how reliable an AI model's prediction is**
using confidence, activation statistics,
gradient statistics, input quality,
and Out-of-Distribution detection.
"""
)

st.divider()

# -----------------------------------------------------
# Load Predictor
# -----------------------------------------------------

@st.cache_resource
def load_predictor():

    return UAIREPredictor()


predictor = load_predictor()

# -----------------------------------------------------
# Upload
# -----------------------------------------------------

uploaded_file = st.file_uploader(

    "Upload an Image",

    type=["png","jpg","jpeg"]

)

# -----------------------------------------------------
# Prediction
# -----------------------------------------------------

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(

        image,

        width=350,

        caption="Uploaded Image"

    )

    with st.spinner("Running UAIRE..."):

        result = predictor.predict(image)

    st.divider()

    col1,col2,col3 = st.columns(3)

    # -----------------------------------------
    # Prediction
    # -----------------------------------------

    with col1:

        st.metric(

            "Prediction",

            result["prediction"]["class"]

        )

        st.metric(

            "Confidence",

            f'{result["prediction"]["confidence"]:.2%}'

        )

    # -----------------------------------------
    # Reliability
    # -----------------------------------------

    with col2:

        st.metric(

            "Reliability Score",

            f'{result["reliability"]["score"]:.2f}%'

        )

        st.metric(

            "Decision",

            result["reliability"]["decision"]

        )

    # -----------------------------------------
    # Failure
    # -----------------------------------------

    with col3:

        st.metric(

            "Failure Probability",

            f'{result["reliability"]["failure_probability"]:.2%}'

        )

    st.divider()

    # -------------------------------------------------
    # Reasons
    # -------------------------------------------------

    st.header("Why?")

    for reason in result["reasons"]:

        st.success(reason)

    st.divider()

    # -------------------------------------------------
    # OOD
    # -------------------------------------------------

    st.header("Out-of-Distribution Analysis")

    col1,col2,col3 = st.columns(3)

    with col1:

        st.metric(

            "MSP",

            result["ood"]["msp"]

        )

    with col2:

        st.metric(

            "Energy",

            result["ood"]["energy"]

        )

    with col3:

        st.metric(

            "Mahalanobis",

            result["ood"]["mahalanobis"]

        )

    st.divider()

    # -------------------------------------------------
    # Signals
    # -------------------------------------------------

    st.header("Reliability Features")

    st.dataframe(

        result["signals"]

    )