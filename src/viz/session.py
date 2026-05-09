"""
Shared session state helpers — model loading, cache, language.
"""

from __future__ import annotations

import logging
import traceback

import streamlit as st

log = logging.getLogger("neuroscope")

from src.inspector.model_inspector import InspectorConfig, ModelInspector
from src.viz.i18n import LANGUAGES, t


SUPPORTED_MODELS = [
    "gpt2",
    "gpt2-medium",
    "EleutherAI/pythia-160m",
    "EleutherAI/pythia-1.4b",
]

_CSS = """
<style>
/* ── Hero (home page) ── */
.ns-hero {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    padding: 1.5rem 0 0.5rem;
}
.ns-hero-icon { font-size: 2.8rem; line-height: 1; }
.ns-hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(120deg, #818CF8 0%, #C084FC 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.ns-hero-sub { color: #94A3B8; font-size: 1rem; margin: 0.25rem 0 0; }

/* ── Feature cards (home page) ── */
.ns-card {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 1rem 1.1rem;
    margin-bottom: 0.8rem;
    transition: border-color 0.15s, box-shadow 0.15s;
}
.ns-card:hover { border-color: #818CF8; box-shadow: 0 0 0 1px #818CF820; }
.ns-card-icon { font-size: 1.6rem; flex-shrink: 0; padding-top: 0.1rem; }
.ns-card-title { font-weight: 600; font-size: 0.95rem; color: #E2E8F0; margin-bottom: 0.2rem; }
.ns-card-desc { color: #94A3B8; font-size: 0.82rem; line-height: 1.5; }

/* ── Page banner ── */
.ns-banner {
    background: linear-gradient(90deg, #1E293B 0%, #0F172A 100%);
    border-left: 4px solid #818CF8;
    border-radius: 0 8px 8px 0;
    padding: 0.9rem 1.4rem;
    margin-bottom: 1.5rem;
}
.ns-banner-title { font-size: 1.6rem; font-weight: 700; color: #E2E8F0; margin: 0 0 0.25rem; }
.ns-banner-desc { color: #94A3B8; font-size: 0.88rem; margin: 0; line-height: 1.5; }

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.8rem 1rem;
}

/* ── Sidebar button full-width ── */
[data-testid="stSidebarContent"] .stButton > button { width: 100%; }
</style>
"""


def inject_global_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def get_lang() -> str:
    return st.session_state.get("lang", "en")


def sidebar_language() -> str:
    lang_name = st.sidebar.selectbox(
        "🌐 Language / Idioma",
        options=list(LANGUAGES.keys()),
        index=0,
        key="lang_selector",
    )
    lang = LANGUAGES[lang_name]
    st.session_state["lang"] = lang
    return lang


def sidebar_model(lang: str) -> ModelInspector | None:
    model_name = st.sidebar.selectbox(
        t("sidebar.model", lang),
        options=SUPPORTED_MODELS,
        index=0,
        key="model_name",
    )

    inspector_key = f"inspector_{model_name}"

    # Already loaded — return cached inspector from session state
    if inspector_key in st.session_state:
        st.sidebar.success(f"✓ {t('sidebar.model_loaded', lang)}: `{model_name}`")
        return st.session_state[inspector_key]  # type: ignore[return-value]

    load_btn = st.sidebar.button(
        f"⚡ Load `{model_name}`",
        key=f"load_btn_{model_name}",
        type="primary",
    )
    if not load_btn:
        return None

    # Load in the main script thread (avoids @st.cache_resource threading conflict on Windows)
    log.info("sidebar_model: loading '%s'", model_name)
    with st.sidebar:
        with st.spinner(t("sidebar.model_loading", lang)):
            try:
                cfg = InspectorConfig(model_name=model_name, device="auto")
                inspector = ModelInspector(cfg)
                inspector.load()
                st.session_state[inspector_key] = inspector
            except Exception as e:
                tb = traceback.format_exc()
                log.error("Failed to load model '%s':\n%s", model_name, tb)
                st.error(f"Failed to load model: {e}")
                st.code(tb, language="python")
                return None

    st.sidebar.success(f"✓ {t('sidebar.model_loaded', lang)}: `{model_name}`")
    return st.session_state[inspector_key]  # type: ignore[return-value]


def sidebar_prompt(lang: str, default: str = "When Mary and John went to the store, John gave a bottle of milk to") -> str:
    return st.sidebar.text_area(
        t("sidebar.prompt", lang),
        value=default,
        height=100,
        key="prompt_input",
    )


def run_forward_pass(inspector: ModelInspector, prompt: str):
    key = f"cache_{inspector.config.model_name}_{hash(prompt)}"
    if key not in st.session_state:
        logits, cache = inspector.run(prompt)
        st.session_state[key] = (logits, cache)
    return st.session_state[key]


def page_header(title: str, intro: str) -> None:
    st.markdown(
        f'<div class="ns-banner">'
        f'<div class="ns-banner-title">{title}</div>'
        f'<div class="ns-banner-desc">{intro}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def require_model(inspector: ModelInspector | None, lang: str) -> bool:
    if inspector is None:
        st.warning(t("error.model_not_loaded", lang))
        return False
    return True
