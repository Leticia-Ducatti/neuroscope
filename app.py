"""
NeuroScope — entry point.
Run with: streamlit run app.py
"""

import os
# Prevents datasets from importing aiohttp, which causes an access violation
# on Python 3.13 / Windows when imported inside Streamlit's ScriptRunner thread.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import streamlit as st

st.set_page_config(
    page_title="NeuroScope",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.viz.session import inject_global_css, sidebar_language, sidebar_model
from src.viz.i18n import t

inject_global_css()

# --- Sidebar ---
lang = sidebar_language()
sidebar_model(lang)

# --- Hero ---
st.markdown("""
<div class="ns-hero">
    <span class="ns-hero-icon">🔬</span>
    <div>
        <div class="ns-hero-title">NeuroScope</div>
        <div class="ns-hero-sub">Mechanistic interpretability for transformer language models</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown(
    "<p style='color:#94A3B8; font-size:0.97rem; line-height:1.7; max-width:760px; margin-bottom:2rem;'>"
    "Inspect the internal computations of GPT-2 and Pythia, reproducing results from "
    "<strong style='color:#C084FC;'>Olsson et al. (2022)</strong> (Induction Heads) and "
    "<strong style='color:#C084FC;'>Wang et al. (2022)</strong> (IOI Circuit). "
    "Load a model in the sidebar, then navigate to any page below."
    "</p>",
    unsafe_allow_html=True,
)

# --- Page cards ---
st.markdown(f"#### {t('home.pages_title', lang)}")

_PAGES = [
    ("🧠", "Attention Explorer",
     "Visualise softmax attention weights for each layer and head."),
    ("📊", "Residual Stream",
     "Track how the residual stream evolves across layers via norm plots and PCA."),
    ("🔭", "Logit Lens",
     "See what token the model predicts at each intermediate layer."),
    ("🔁", "Induction Heads",
     "Detect induction heads using prefix matching and copying scores."),
    ("🔗", "IOI Circuit",
     "Map attention head contributions to indirect object identification via activation patching."),
]

col1, col2 = st.columns(2)
for i, (icon, title, desc) in enumerate(_PAGES):
    with (col1 if i % 2 == 0 else col2):
        st.markdown(
            f'<div class="ns-card">'
            f'<span class="ns-card-icon">{icon}</span>'
            f'<div><div class="ns-card-title">{title}</div>'
            f'<div class="ns-card-desc">{desc}</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()
st.caption("NeuroScope v0.1 · GPT-2 small · Olsson et al. (2022) · Wang et al. (2022)")
