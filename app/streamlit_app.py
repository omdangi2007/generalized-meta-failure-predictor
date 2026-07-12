import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import streamlit as st
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

st.divider()