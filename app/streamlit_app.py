import json
import os
import sys
from html import escape
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image, ImageStat

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.inference.predictor import UAIREPredictor
from src.models.model_loader import load_resnet18_cifar10
from src.xai.gradcam_features import GradCAMExplainer
from src.xai.lime_features import LIMEImageExplainer
from src.xai.shap_features import SHAPMetaExplainer


st.set_page_config(
    page_title="UAIRE Reliability Platform",
    page_icon="U",
    layout="wide",
    initial_sidebar_state="expanded",
)


SIGNAL_GROUPS = {
    "Prediction": ["confidence", "entropy", "margin", "msp", "probability"],
    "Activation": ["activation", "neuron", "sparsity", "positive"],
    "Gradient": ["gradient", "grad"],
    "OOD": ["ood", "energy", "mahalanobis", "distance"],
    "Input Quality": ["brightness", "contrast", "blur", "laplacian", "edge", "image_entropy"],
}

PIPELINE_STAGES = [
    ("01", "Input Intake", "Image is normalized and prepared for audit."),
    ("02", "Backbone Prediction", "The selected neural model produces class evidence."),
    ("03", "Neural State Capture", "Activations and gradients are collected from the model."),
    ("04", "Signal Extraction", "Confidence, behaviour and image-quality signals are fused."),
    ("05", "OOD Analysis", "Energy, MSP and feature-distance cues are evaluated."),
    ("06", "Meta-Failure Model", "A trained meta-model estimates prediction failure risk."),
    ("07", "Reliability Scoring", "Failure risk is converted into a trust score and decision."),
    ("08", "Explanation Layer", "Human-readable reasons and exportable audit data are generated."),
]


def load_css() -> None:
    css_path = PROJECT_ROOT / "app" / "assets" / "style.css"
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


def clean_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: clean_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_value(item) for item in value]
    return value


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def format_signal_name(name: str) -> str:
    return name.replace("_", " ").replace("-", " ").title()


def decision_tone(decision: str) -> str:
    normalized = decision.lower()
    if "highly" in normalized or normalized == "reliable":
        return "good"
    if "caution" in normalized:
        return "warn"
    return "bad"


def signal_group(name: str) -> str:
    lowered = name.lower()
    for group, keywords in SIGNAL_GROUPS.items():
        if any(keyword in lowered for keyword in keywords):
            return group
    return "Other"


def visual_score(values: list[float]) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=float)
    arr = np.nan_to_num(np.abs(arr), nan=0.0, posinf=0.0, neginf=0.0)
    high = float(np.percentile(arr, 90)) if arr.size else 0.0
    if high <= 0:
        return 0.0
    return float(np.clip(np.mean(np.clip(arr / high, 0, 1)) * 100, 0, 100))


def normalize_series(series: pd.Series) -> pd.Series:
    values = series.astype(float).abs()
    low = values.min()
    high = values.max()
    if high == low:
        return pd.Series(np.ones(len(values)) * 50, index=values.index)
    return ((values - low) / (high - low) * 100).clip(0, 100)


def signal_dataframe(signals: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for name, value in signals.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
            rows.append(
                {
                    "Group": signal_group(name),
                    "Signal": format_signal_name(name),
                    "Raw Name": name,
                    "Value": float(value),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["Group", "Signal", "Raw Name", "Value", "Visual Score"])
    df = pd.DataFrame(rows).sort_values(["Group", "Signal"]).reset_index(drop=True)
    df["Visual Score"] = df.groupby("Group")["Value"].transform(normalize_series)
    return df


def grouped_summary(signal_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group in SIGNAL_GROUPS:
        subset = signal_df[signal_df["Group"] == group]
        rows.append(
            {
                "Dimension": group,
                "Signals": int(len(subset)),
                "Visual Strength": float(subset["Visual Score"].mean()) if len(subset) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def image_statistics(image: Image.Image, signals: dict[str, Any]) -> dict[str, Any]:
    rgb = image.convert("RGB")
    gray = image.convert("L")
    stat = ImageStat.Stat(rgb)
    width, height = rgb.size
    return {
        "Resolution": f"{width} x {height}",
        "Aspect Ratio": f"{width / max(height, 1):.2f}:1",
        "Mean RGB": ", ".join(f"{value:.1f}" for value in stat.mean),
        "Brightness": signals.get("brightness"),
        "Contrast": signals.get("contrast"),
        "Image Entropy": signals.get("image_entropy"),
        "Blur Score": signals.get("blur_score"),
        "Edge Density": signals.get("edge_density"),
        "Pixel Range": f"{min(gray.getextrema())} - {max(gray.getextrema())}",
    }


def html_card(title: str, value: str, caption: str = "", tone: str = "neutral") -> None:
    st.markdown(
        f"""
        <div class="uaire-card metric-card tone-{tone}">
            <div class="metric-label">{escape(title)}</div>
            <div class="metric-value">{escape(value)}</div>
            <div class="metric-caption">{escape(caption)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_header(label: str, title: str, body: str = "") -> None:
    st.markdown(
        f"""
        <div class="section-header">
            <p class="eyebrow">{escape(label)}</p>
            <h2>{escape(title)}</h2>
            <span>{escape(body)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def trust_decision_card(score: float, failure_probability: float, decision: str, prediction: dict[str, Any]) -> None:
    tone = decision_tone(decision)
    st.markdown(
        f"""
        <div class="trust-hero uaire-card tone-{tone}">
            <div class="gauge-shell">
                <div class="score-ring animated" style="--score:{score:.2f};">
                    <div>
                        <span>{score:.1f}%</span>
                        <small>Reliability</small>
                    </div>
                </div>
            </div>
            <div class="trust-copy">
                <p class="eyebrow">Trust Decision</p>
                <h2>{escape(decision)}</h2>
                <p>
                    UAIRE audits the model prediction as <strong>{escape(prediction["class"])}</strong>
                    with {float(prediction["confidence"]) * 100:.2f}% backbone confidence.
                </p>
                <div class="risk-bar">
                    <div style="width:{failure_probability * 100:.2f}%"></div>
                </div>
                <small>{failure_probability * 100:.2f}% estimated failure probability from the meta-failure predictor</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pipeline_timeline() -> None:
    html = "".join(
        (
            f'<div class="timeline-step"><span>{number}</span><div>'
            f'<strong>{escape(title)}</strong><p>{escape(description)}</p>'
            f'</div></div>'
        )
        for number, title, description in PIPELINE_STAGES
    )
    st.markdown(f'<div class="timeline">{html}</div>', unsafe_allow_html=True)


def fingerprint_visual(summary_df: pd.DataFrame) -> None:
    rows = []
    for row in summary_df.itertuples(index=False):
        strength = float(row[2])
        width = max(4, strength)
        rows.append(
            (
                f'<div class="fingerprint-row"><div>'
                f'<strong>{escape(row.Dimension)}</strong><span>{int(row.Signals)} signals</span>'
                f'</div><div class="fingerprint-track"><span style="width:{width:.2f}%"></span></div>'
                f'<em>{strength:.1f}</em></div>'
            )
        )
    st.markdown(f'<div class="uaire-card fingerprint-card">{"".join(rows)}</div>', unsafe_allow_html=True)


def radar_chart(summary_df: pd.DataFrame) -> go.Figure:
    theta = summary_df["Dimension"].tolist()
    values = summary_df["Visual Strength"].round(2).tolist()
    if theta:
        theta = theta + [theta[0]]
        values = values + [values[0]]
    fig = go.Figure(
        data=[
            go.Scatterpolar(
                r=values,
                theta=theta,
                fill="toself",
                line_color="#69a7ff",
                fillcolor="rgba(105, 167, 255, 0.22)",
                name="Neural Behaviour Signature",
            )
        ]
    )
    fig.update_layout(
        height=380,
        margin=dict(l=28, r=28, t=28, b=28),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dfe7f2"),
        polar=dict(
            bgcolor="rgba(255,255,255,0.02)",
            radialaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.12)", tickfont=dict(size=10)),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.10)"),
        ),
        showlegend=False,
    )
    return fig


def grouped_signal_chart(signal_df: pd.DataFrame) -> go.Figure:
    chart_df = signal_df[signal_df["Group"].isin(SIGNAL_GROUPS)].copy()
    chart_df = chart_df.sort_values(["Group", "Visual Score"], ascending=[True, False])
    chart_df = chart_df.groupby("Group").head(5)
    fig = go.Figure()
    for group in SIGNAL_GROUPS:
        subset = chart_df[chart_df["Group"] == group]
        if subset.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=subset["Visual Score"],
                y=subset["Signal"],
                orientation="h",
                name=group,
                hovertemplate="%{y}<br>Visual score: %{x:.1f}<extra></extra>",
            )
        )
    fig.update_layout(
        height=440,
        barmode="group",
        margin=dict(l=12, r=12, t=16, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dfe7f2"),
        xaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.08)", title="Normalized visual score"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.04)", automargin=True),
        legend=dict(orientation="h", y=-0.22),
    )
    return fig


def generate_gradcam(predictor: UAIREPredictor, image: Image.Image, class_index: int) -> dict[str, Any]:
    explainer = GradCAMExplainer(
        predictor.model,
        predictor.target_layer,
        predictor.device,
    )
    return explainer.explain(
        image,
        class_index=class_index,
    )


def generate_lime(predictor: UAIREPredictor, image: Image.Image, class_index: int) -> dict[str, Any]:
    explainer = LIMEImageExplainer(
        predictor.model,
        predictor.device,
    )
    return explainer.explain(
        image,
        class_index=class_index,
        num_samples=160,
        num_features=8,
        batch_size=32,
        max_side=160,
    )


def global_shap_importance(shap_result: dict[str, Any]) -> pd.DataFrame:
    explanation = shap_result["background_explanation"]
    values = np.asarray(explanation.values)
    importance = np.abs(values).mean(axis=0)
    frame = pd.DataFrame(
        {
            "Feature": explanation.feature_names,
            "Mean |SHAP|": importance,
        }
    )
    return frame.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)


def shap_waterfall_figure(shap_result: dict[str, Any]):
    import matplotlib.pyplot as plt
    import shap

    plt.close("all")
    shap.plots.waterfall(
        shap_result["local_explanation"][0],
        max_display=12,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(9.5, 5.4)
    return fig


def shap_beeswarm_figure(shap_result: dict[str, Any]):
    import matplotlib.pyplot as plt
    import shap

    plt.close("all")
    shap.plots.beeswarm(
        shap_result["background_explanation"],
        max_display=15,
        show=False,
    )
    fig = plt.gcf()
    fig.set_size_inches(9.5, 5.8)
    return fig


def shap_explanation_card(shap_result: dict[str, Any]) -> None:
    top_features = shap_result["top_features"].head(10)
    rows = []
    for row in top_features.itertuples(index=False):
        tone = "risk" if row.shap_value > 0 else "protective"
        rows.append(
            (
                f'<div class="shap-row {tone}"><div>'
                f'<strong>{escape(format_signal_name(row.feature))}</strong>'
                f'<span>{escape(row.direction)} | raw value {float(row.value):.4f}</span>'
                f'</div><em>{float(row.shap_value):+.4f}</em></div>'
            )
        )

    st.markdown(
        (
            '<div class="uaire-card shap-summary-card">'
            '<p class="eyebrow">Local SHAP Explanation</p>'
            f'<h3>{escape(shap_result["summary"])}</h3>'
            '<div class="shap-scoreline">'
            f'<span>Background failure probability: {shap_result["base_failure_probability"]:.3f}</span>'
            f'<span>Explained failure probability: {shap_result["failure_probability"]:.3f}</span>'
            '</div>'
            f'<div class="shap-list">{"".join(rows)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_predictor() -> UAIREPredictor:
    model = load_resnet18_cifar10()
    return UAIREPredictor(model=model)


@st.cache_resource
def load_shap_explainer() -> SHAPMetaExplainer:
    return SHAPMetaExplainer(background_rows=256)


load_css()

with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-mark">U</div>
            <div>
                <h1>UAIRE</h1>
                <p>Universal AI Reliability Engine</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader("Upload image for audit", type=["png", "jpg", "jpeg"])
    model_name = st.selectbox("Backbone model", ["ResNet18 CIFAR10"], index=0)
    mode = st.radio("Dashboard mode", ["Executive", "Advanced"], horizontal=True)

st.markdown(
    """
    <section class="hero platform-hero">
        <div>
            <p class="eyebrow">AI auditing framework, not an image classifier</p>
            <h1>UAIRE Reliability Platform</h1>
            <p class="hero-copy">
                A model-agnostic reliability layer that evaluates whether a neural prediction
                should be trusted using behavioural signals, OOD evidence and meta-failure prediction.
            </p>
        </div>
        <div class="hero-stats">
            <span>Trust scoring</span>
            <span>Neural behaviour signature</span>
            <span>Reliability fingerprint</span>
        </div>
    </section>
    """,
    unsafe_allow_html=True,
)

if uploaded_file is None:
    section_header(
        "Executive Summary",
        "Upload an image to generate an AI reliability audit.",
        "The dashboard will produce a trust decision, grouped signal evidence, a reliability fingerprint and an exportable report.",
    )
    st.markdown(
        """
        <div class="empty-state">
            <p class="eyebrow">Awaiting audit input</p>
            <h2>UAIRE evaluates whether a model prediction deserves trust.</h2>
            <p>
                The pipeline keeps the classifier in place, then audits its prediction
                using confidence, activations, gradients, input quality, OOD geometry and SHAP explanations.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    section_header("AI Reasoning Timeline", "How UAIRE audits a prediction")
    pipeline_timeline()
    st.stop()

predictor = load_predictor()
image = Image.open(uploaded_file).convert("RGB")

with st.spinner("Running UAIRE reliability audit..."):
    result = predictor.predict(image)

prediction = result["prediction"]
reliability = result["reliability"]
ood = result["ood"]
signals = result["signals"]
reasons = result["reasons"]

score = float(reliability["score"])
failure_probability = float(reliability["failure_probability"])
confidence = float(prediction["confidence"])
decision = reliability["decision"]
tone = decision_tone(decision)
signal_df = signal_dataframe(signals)
summary_df = grouped_summary(signal_df)
image_stats = image_statistics(image, signals)

try:
    shap_result = load_shap_explainer().explain(signals)
    shap_error = None
except Exception as exc:
    shap_result = None
    shap_error = str(exc)

try:
    gradcam_result = generate_gradcam(
        predictor,
        image,
        int(prediction["class_index"]),
    )
    gradcam_error = None
except Exception as exc:
    gradcam_result = None
    gradcam_error = str(exc)

try:
    lime_result = generate_lime(
        predictor,
        image,
        int(prediction["class_index"]),
    )
    lime_error = None
except Exception as exc:
    lime_result = None
    lime_error = str(exc)

report = {
    "report_type": "UAIRE AI Reliability Report",
    "report_status": "JSON active with SHAP explanations, PDF-ready architecture planned",
    "model": model_name,
    "prediction": clean_value(prediction),
    "reliability": clean_value(reliability),
    "ood": clean_value(ood),
    "image_statistics": clean_value(image_stats),
    "signal_groups": clean_value(summary_df.to_dict(orient="records")),
    "signals": clean_value(signals),
    "reasons": clean_value(reasons),
    "shap": None if shap_result is None else {
        "natural_language_summary": shap_result["summary"],
        "base_failure_probability": shap_result["base_failure_probability"],
        "explained_failure_probability": shap_result["failure_probability"],
        "top_10_features": clean_value(
            shap_result["top_features"].to_dict(orient="records")
        ),
    },
    "gradcam": None if gradcam_result is None else {
        "status": "generated",
        "class_index": gradcam_result["class_index"],
        "class_name": prediction["class"],
        "confidence": gradcam_result["confidence"],
        "interpretation": "GradCAM explains where the backbone model looked before UAIRE audited whether the prediction should be trusted.",
    },
    "lime": None if lime_result is None else {
        "status": "generated",
        "class_index": lime_result["class_index"],
        "class_name": prediction["class"],
        "confidence": lime_result["confidence"],
        "top_superpixels": clean_value(lime_result["top_superpixels"]),
        "interpretation": "LIME explains which image superpixels locally supported or contradicted the backbone prediction.",
    },
}

section_header(
    "Executive Summary",
    "Prediction trust audit",
    "UAIRE has converted model behaviour into a presentation-ready reliability decision.",
)

summary_cols = st.columns(5, gap="medium")
with summary_cols[0]:
    html_card("Trust Decision", decision, "Final UAIRE audit verdict", tone)
with summary_cols[1]:
    html_card("Reliability", f"{score:.2f}%", "Meta-failure score inversion", tone)
with summary_cols[2]:
    html_card("Failure Risk", percent(failure_probability), "Estimated probability of failure", "bad" if failure_probability >= 0.5 else "good")
with summary_cols[3]:
    html_card("Prediction", prediction["class"], f"Backbone class index {prediction['class_index']}")
with summary_cols[4]:
    html_card("Confidence", percent(confidence), "Backbone confidence only", "good" if confidence >= 0.75 else "warn")

top_left, top_right = st.columns([1.35, 0.65], gap="large")
with top_left:
    trust_decision_card(score, failure_probability, decision, prediction)
with top_right:
    st.download_button(
        "Download AI Reliability Report",
        data=json.dumps(report, indent=2).encode("utf-8"),
        file_name="uaire_ai_reliability_report.json",
        mime="application/json",
        width="stretch",
    )
    st.markdown(
        """
        <div class="report-note">
            <strong>Report architecture</strong>
            <span>JSON export is active now. The report payload is structured for a future PDF dossier without retraining or pipeline changes.</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

section_header(
    "AI Reasoning Timeline",
    "End-to-end audit path",
    "Each stage is a reliability operation layered beside the neural network, not a replacement for the classifier.",
)
pipeline_timeline()

section_header(
    "Input Evidence",
    "Uploaded image, input-quality audit and visual explanations",
    "GradCAM shows where the backbone looked; LIME shows which superpixels affected the local prediction.",
)
image_col, stat_col = st.columns([0.74, 1.26], gap="large")
with image_col:
    st.markdown('<div class="panel-title">Input Artifact</div>', unsafe_allow_html=True)
    st.image(image, width="stretch")
    st.caption(f"Backbone: {model_name}")
with stat_col:
    stat_cols = st.columns(3, gap="medium")
    stat_items = [
        ("Resolution", image_stats["Resolution"], "Original upload dimensions"),
        ("Brightness", image_stats["Brightness"], "Input quality signal"),
        ("Contrast", image_stats["Contrast"], "Input quality signal"),
        ("Entropy", image_stats["Image Entropy"], "Image complexity"),
        ("Blur Score", image_stats["Blur Score"], "Laplacian variance"),
        ("Edge Density", image_stats["Edge Density"], "Structural detail"),
    ]
    for idx, (label, value, caption) in enumerate(stat_items):
        display = "Unavailable" if value is None else (value if isinstance(value, str) else f"{float(value):.4f}")
        with stat_cols[idx % 3]:
            html_card(label, display, caption)

section_header(
    "Backbone Attention vs UAIRE Reliability",
    "GradCAM and LIME visual explanations",
    "Attention and local superpixel evidence explain the backbone prediction; UAIRE separately audits whether that prediction should be trusted.",
)
st.markdown(
    (
        '<div class="uaire-card gradcam-explainer">'
        '<div><p class="eyebrow">Backbone Prediction</p>'
        f'<h3>{escape(prediction["class"])} | {confidence * 100:.2f}% confidence</h3>'
        '<p>GradCAM and LIME explain the visual evidence behind this class score.</p></div>'
        '<div><p class="eyebrow">UAIRE Reliability</p>'
        f'<h3>{escape(decision)} | {score:.2f}% reliability</h3>'
        '<p>UAIRE audits whether the visually explained prediction should be trusted.</p></div>'
        '</div>'
    ),
    unsafe_allow_html=True,
)

xai_cols = st.columns(4, gap="medium")
with xai_cols[0]:
    st.markdown('<div class="panel-title">Original Image</div>', unsafe_allow_html=True)
    st.image(image, width="stretch")
with xai_cols[1]:
    st.markdown('<div class="panel-title">GradCAM Overlay</div>', unsafe_allow_html=True)
    if gradcam_result is None:
        st.error(f"GradCAM unavailable: {gradcam_error}")
    else:
        st.image(gradcam_result["overlay"], width="stretch")
with xai_cols[2]:
    st.markdown('<div class="panel-title">LIME Superpixels</div>', unsafe_allow_html=True)
    if lime_result is None:
        st.error(f"LIME unavailable: {lime_error}")
    else:
        st.image(lime_result["positive_overlay"], width="stretch")
with xai_cols[3]:
    st.markdown('<div class="panel-title">LIME Signed Regions</div>', unsafe_allow_html=True)
    if lime_result is None:
        st.error(f"LIME unavailable: {lime_error}")
    else:
        st.image(lime_result["signed_overlay"], width="stretch")

comparison_cols = st.columns([1, 1], gap="medium")
with comparison_cols[0]:
    st.markdown(
        """
        <div class="uaire-card xai-compare-card">
            <p class="eyebrow">GradCAM</p>
            <h3>Backbone attention map</h3>
            <p>Highlights spatial regions with strong gradient-weighted activation for the predicted class.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with comparison_cols[1]:
    top_lime = [] if lime_result is None else lime_result["top_superpixels"][:5]
    lime_rows = "".join(
        (
            f'<div class="lime-row"><strong>Superpixel {item["superpixel"]}</strong>'
            f'<span>{escape(item["direction"])} | weight {item["weight"]:+.4f}</span></div>'
        )
        for item in top_lime
    )
    st.markdown(
        (
            '<div class="uaire-card xai-compare-card">'
            '<p class="eyebrow">LIME</p>'
            '<h3>Local superpixel evidence</h3>'
            '<p>Perturbs image regions to estimate which superpixels support or contradict the backbone class.</p>'
            f'<div class="lime-list">{lime_rows}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )

section_header(
    "Reliability Fingerprint",
    "Grouped signal evidence",
    "Values below are normalized for visualization only; raw feature values remain available in the advanced table and report.",
)
fingerprint_col, radar_col = st.columns([1, 1], gap="large")
with fingerprint_col:
    fingerprint_visual(summary_df)
with radar_col:
    st.plotly_chart(radar_chart(summary_df), width="stretch", config={"displayModeBar": False})

section_header(
    "Grouped Reliability Signals",
    "Prediction, activation, gradient, OOD and input-quality dimensions",
    "This view avoids raw mixed feature dumps and presents reliability evidence as audit categories.",
)
if not signal_df.empty:
    st.plotly_chart(grouped_signal_chart(signal_df), width="stretch", config={"displayModeBar": False})
else:
    st.info("No numeric reliability signals were returned by the current extractor set.")

section_header(
    "Explainable AI",
    "SHAP explanation for the meta-failure predictor",
    "These explanations describe why the Random Forest meta-model moved toward Reliable or Failure for this prediction.",
)
if shap_result is None:
    st.error(
        "SHAP explanations could not be generated. "
        f"Reason: {shap_error}"
    )
else:
    shap_explanation_card(shap_result)

    shap_tabs = st.tabs(
        [
            "Top 10 Features",
            "Waterfall Plot",
            "Beeswarm Plot",
            "Global Importance",
        ]
    )

    with shap_tabs[0]:
        top_display = shap_result["top_features"].head(10).copy()
        top_display["feature"] = top_display["feature"].map(format_signal_name)
        st.dataframe(
            top_display.rename(
                columns={
                    "feature": "Reliability Feature",
                    "value": "Raw Value",
                    "shap_value": "SHAP Value",
                    "impact": "|SHAP|",
                    "direction": "Interpretation",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Raw Value": st.column_config.NumberColumn(format="%.6f"),
                "SHAP Value": st.column_config.NumberColumn(format="%+.6f"),
                "|SHAP|": st.column_config.NumberColumn(format="%.6f"),
            },
        )

    with shap_tabs[1]:
        st.pyplot(
            shap_waterfall_figure(shap_result),
            clear_figure=True,
        )
        st.caption(
            "Local waterfall plot for class 1: failure probability. Positive values push toward Failure; negative values support Reliable."
        )

    with shap_tabs[2]:
        st.pyplot(
            shap_beeswarm_figure(shap_result),
            clear_figure=True,
        )
        st.caption(
            "Global beeswarm plot computed from the SHAP background sample aligned to UAIRE's saved feature order."
        )

    with shap_tabs[3]:
        importance_df = global_shap_importance(shap_result).head(15)
        st.bar_chart(
            importance_df.set_index("Feature")["Mean |SHAP|"],
            height=360,
        )
        st.dataframe(
            importance_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Mean |SHAP|": st.column_config.NumberColumn(format="%.6f"),
            },
        )

section_header("Reasoning Output", "Why UAIRE made this trust decision")
reason_cols = st.columns(2, gap="medium")
for index, reason in enumerate(reasons or ["No dominant risk reason was generated."]):
    with reason_cols[index % 2]:
        st.markdown(
            (
                f'<div class="uaire-card reason-card">'
                f'<span>{index + 1:02d}</span><p>{escape(reason)}</p>'
                f'</div>'
            ),
            unsafe_allow_html=True,
        )

if mode == "Advanced":
    section_header("Advanced Workspace", "Raw audit data for researchers")
    tab_signals, tab_ood, tab_report = st.tabs(["Grouped Signal Registry", "OOD Evidence", "Report Payload"])

    with tab_signals:
        st.dataframe(
            signal_df,
            width="stretch",
            hide_index=True,
            column_config={
                "Value": st.column_config.NumberColumn(format="%.6f"),
                "Visual Score": st.column_config.ProgressColumn(format="%.1f", min_value=0, max_value=100),
            },
        )

    with tab_ood:
        ood_df = pd.DataFrame(
            [
                {"Metric": "MSP", "Value": ood.get("msp"), "Meaning": "Maximum softmax probability"},
                {"Metric": "Energy", "Value": ood.get("energy"), "Meaning": "Normalized energy score"},
                {"Metric": "Mahalanobis", "Value": ood.get("mahalanobis"), "Meaning": "Feature-space distance"},
            ]
        )
        st.dataframe(ood_df, width="stretch", hide_index=True)

    with tab_report:
        st.json(report)
