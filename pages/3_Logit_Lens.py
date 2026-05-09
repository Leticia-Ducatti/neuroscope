import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.viz.session import inject_global_css, sidebar_language, sidebar_model, sidebar_prompt, run_forward_pass, page_header, require_model
from src.viz.i18n import t

st.set_page_config(page_title="Logit Lens — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
inspector = sidebar_model(lang)

page_header(t("logit_lens.title", lang), t("logit_lens.intro", lang))

if not require_model(inspector, lang):
    st.stop()

prompt = sidebar_prompt(lang)
top_k = st.sidebar.slider(t("sidebar.top_k", lang), 1, 20, 5, key="ll_topk")
run_btn = st.sidebar.button(t("sidebar.run", lang), type="primary")

if run_btn or "ll_cache" in st.session_state:
    if run_btn:
        with st.spinner(t("sidebar.running", lang)):
            logits, cache = run_forward_pass(inspector, prompt)
            st.session_state["ll_cache"] = (logits, cache)
            st.session_state["ll_prompt"] = prompt
    else:
        logits, cache = st.session_state["ll_cache"]
        prompt = st.session_state.get("ll_prompt", prompt)

    lens = inspector.get_logit_lens(cache, top_k=top_k)
    n_layers = inspector.n_layers
    layer_labels = ["Embed"] + [f"L{i}" for i in range(n_layers)]

    final_token = lens[-1][0][0].strip() if lens else ""

    rows = []
    for layer_i, (label, layer_preds) in enumerate(zip(layer_labels, lens)):
        for rank, (token, prob) in enumerate(layer_preds):
            rows.append({
                t("logit_lens.col_layer", lang): label,
                t("logit_lens.col_rank", lang): rank + 1,
                t("logit_lens.col_token", lang): repr(token),
                t("logit_lens.col_prob", lang): round(prob, 4),
            })
    df = pd.DataFrame(rows)

    st.subheader(t("logit_lens.table_title", lang, k=top_k))

    top1_probs = [preds[0][1] for preds in lens]
    top1_tokens = [repr(preds[0][0]) for preds in lens]

    fig = px.bar(
        x=layer_labels,
        y=top1_probs,
        text=top1_tokens,
        labels={"x": t("logit_lens.col_layer", lang), "y": t("logit_lens.col_prob", lang)},
        title=t("logit_lens.table_title", lang, k=1),
        color=top1_probs,
        color_continuous_scale="Purples",
        template="plotly_dark",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        coloraxis_showscale=False,
        height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1E293B",
    )
    st.plotly_chart(fig, use_container_width=True)

    convergence_layer = None
    for i, preds in enumerate(lens):
        if preds[0][0].strip() == final_token:
            convergence_layer = layer_labels[i]
            break
    if convergence_layer:
        st.info(f"{t('logit_lens.convergence', lang)}: **{convergence_layer}** → `{repr(final_token)}`")

    with st.expander(f"Full table — top-{top_k} per layer"):
        st.dataframe(df, use_container_width=True, hide_index=True)

else:
    st.info(t("error.no_cache", lang))
