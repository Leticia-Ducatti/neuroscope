import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px

from src.viz.session import inject_global_css, sidebar_language, sidebar_model, sidebar_prompt, run_forward_pass, page_header, require_model
from src.viz.i18n import t

st.set_page_config(page_title="Attention Explorer — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
inspector = sidebar_model(lang)

page_header(t("attention.title", lang), t("attention.intro", lang))

if not require_model(inspector, lang):
    st.stop()

prompt = sidebar_prompt(lang)
run_btn = st.sidebar.button(t("sidebar.run", lang), type="primary")

if run_btn or "attn_cache" in st.session_state:
    if run_btn:
        with st.spinner(t("sidebar.running", lang)):
            logits, cache = run_forward_pass(inspector, prompt)
            st.session_state["attn_cache"] = (logits, cache)
            st.session_state["attn_prompt"] = prompt
    else:
        logits, cache = st.session_state["attn_cache"]
        prompt = st.session_state.get("attn_prompt", prompt)

    str_tokens = inspector.to_str_tokens(prompt)
    n_layers = inspector.n_layers
    n_heads = inspector.n_heads

    layer = st.sidebar.slider(t("sidebar.layer", lang), 0, n_layers - 1, 0, key="attn_layer")
    selected_heads = st.sidebar.multiselect(
        t("attention.select_heads", lang),
        options=list(range(n_heads)),
        default=list(range(min(4, n_heads))),
        format_func=lambda h: f"Head {h}",
        key="attn_heads",
    )
    if not selected_heads:
        selected_heads = list(range(n_heads))

    patterns = inspector.get_attention_patterns(cache, layer)  # [n_heads, seq, seq]

    cols = st.columns(min(len(selected_heads), 3))
    for i, head in enumerate(selected_heads):
        col = cols[i % len(cols)]
        with col:
            pattern = patterns[head].cpu().float().numpy()
            fig = px.imshow(
                pattern,
                x=str_tokens,
                y=str_tokens,
                title=t("attention.heatmap_title", lang, layer=layer, head=head),
                labels={
                    "x": t("attention.axis_key", lang),
                    "y": t("attention.axis_query", lang),
                    "color": "Attention",
                },
                color_continuous_scale="Purples",
                aspect="auto",
                zmin=0,
                zmax=1,
                template="plotly_dark",
            )
            fig.update_layout(
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="#1E293B",
            )
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info(t("error.no_cache", lang))
