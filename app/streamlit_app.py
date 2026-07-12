import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
import pandas as pd
from PIL import Image

from src.inference.predictor import UAIREPredictor

# ----------------------------------------------------------
# Page Config
# ----------------------------------------------------------

st.set_page_config(
    page_title="UAIRE",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Custom CSS
# ----------------------------------------------------------

st.markdown("""
<style>

.main{
    background:#0E1117;
}

.block-container{
    padding-top:2rem;
}

.metric-card{
    background:#1A1D24;
    border-radius:15px;
    padding:20px;
    text-align:center;
    border:1px solid #2E3440;
}

.metric-title{
    font-size:18px;
    color:#AAB2BF;
}

.metric-value{
    font-size:32px;
    font-weight:bold;
    color:white;
}

.section-title{
    font-size:26px;
    font-weight:bold;
    color:white;
    margin-top:20px;
}

.reason-card{
    background:#182028;
    padding:12px;
    border-radius:12px;
    margin-bottom:10px;
    border-left:6px solid #22C55E;
}

.status-card{
    background:#182028;
    padding:15px;
    border-radius:12px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------

st.sidebar.title("🛡️ UAIRE")

st.sidebar.markdown(
"""
Universal AI Reliability Engine
"""
)

st.sidebar.divider()

uploaded_file = st.sidebar.file_uploader(
    "📤 Upload Image",
    type=["png","jpg","jpeg"]
)

model_name = st.sidebar.selectbox(
    "🧠 Backbone",
    [
        "ResNet18 (Supported)"
    ]
)

mode = st.sidebar.radio(
    "📊 Analysis Mode",
    [
        "Basic",
        "Advanced"
    ]
)

st.sidebar.divider()

st.sidebar.success("✅ AI Model Ready")
st.sidebar.success("✅ Meta Model Ready")
st.sidebar.success("✅ Reliability Engine Ready")
st.sidebar.success("✅ OOD Detector Ready")

# ----------------------------------------------------------
# Load Predictor
# ----------------------------------------------------------

@st.cache_resource
def load_predictor():

    return UAIREPredictor()

predictor = load_predictor()

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.title("🛡️ UAIRE Dashboard")

st.caption(
    "Universal AI Reliability Engine"
)
# ==========================================================
# Image Upload & Prediction
# ==========================================================

st.divider()

if uploaded_file is None:

    st.info("👈 Upload an image from the sidebar.")

    st.stop()

image = Image.open(uploaded_file).convert("RGB")

left, right = st.columns([1, 1])

with left:

    st.subheader("🖼 Uploaded Image")

    st.image(
        image,
        use_container_width=True
    )

with right:

    st.subheader("🧠 UAIRE Analysis")

    with st.spinner("Running UAIRE..."):

        result = predictor.predict(image)
# ==========================================================
# Prediction Metrics
# ==========================================================

st.divider()

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "Prediction",

        result["prediction"]["class"]

    )

with col2:

    st.metric(

        "Confidence",

        f"{result['prediction']['confidence']*100:.2f}%"

    )

with col3:

    st.metric(

        "Reliability",

        f"{result['reliability']['score']:.2f}%"

    )

with col4:

    st.metric(

        "Failure Risk",

        f"{result['reliability']['failure_probability']*100:.2f}%"

    )

# ==========================================================
# Decision
# ==========================================================

st.divider()

st.subheader("🛡 Reliability Decision")

decision = result["reliability"]["decision"]

if decision == "Highly Reliable":

    st.success(decision)

elif decision == "Reliable":

    st.success(decision)

elif decision == "Use With Caution":

    st.warning(decision)

else:

    st.error(decision)


# ==========================================================
# Reasons
# ==========================================================

st.divider()

st.subheader("💡 Why did UAIRE trust this prediction?")

for reason in result["reasons"]:

    st.success(reason)

# ==========================================================
# OOD Metrics
# ==========================================================

st.divider()

st.subheader("🌍 Distribution Analysis")

o1, o2, o3 = st.columns(3)

with o1:

    st.metric(

        "MSP",

        f"{result['ood']['msp']:.4f}"

        if result["ood"]["msp"] is not None

        else "N/A"

    )

with o2:

    st.metric(

        "Energy",

        f"{result['ood']['energy']:.4f}"

        if result["ood"]["energy"] is not None

        else "N/A"

    )

with o3:

    st.metric(

        "Mahalanobis",

        f"{result['ood']['mahalanobis']:.4f}"

        if result["ood"]["mahalanobis"] is not None

        else "N/A"

    )
# ==========================================================
# Advanced Mode
# ==========================================================

if mode == "Advanced":

    st.divider()

    st.subheader("📊 Reliability Features")

    feature_df = pd.DataFrame(

        result["signals"],

        index=[0]

    )

    st.dataframe(

        feature_df,

        use_container_width=True

    )