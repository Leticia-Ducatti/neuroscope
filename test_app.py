import streamlit as st
st.set_page_config(page_title="Test", layout="wide")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from src.viz.i18n import LANGUAGES
from src.viz.session import sidebar_language, sidebar_model, require_model

lang = sidebar_language()
inspector = sidebar_model(lang)

st.write("App render OK")

if inspector is None:
    st.info("Click '⚡ Load gpt2' in the sidebar.")
else:
    st.success(f"Model loaded: {inspector.config.model_name} — layers={inspector.n_layers}")
