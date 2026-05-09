import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np

from src.viz.session import inject_global_css, sidebar_language, sidebar_model, sidebar_prompt, run_forward_pass, page_header, require_model
from src.viz.i18n import t

st.set_page_config(page_title="Residual Stream — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
inspector = sidebar_model(lang)

page_header(t("residual.title", lang), t("residual.intro", lang))

if not require_model(inspector, lang):
    st.stop()

prompt = sidebar_prompt(lang)
run_btn = st.sidebar.button(t("sidebar.run", lang), type="primary")

if run_btn or "resid_cache" in st.session_state:
    if run_btn:
        with st.spinner(t("sidebar.running", lang)):
            logits, cache = run_forward_pass(inspector, prompt)
            st.session_state["resid_cache"] = (logits, cache)
            st.session_state["resid_prompt"] = prompt
    else:
        logits, cache = st.session_state["resid_cache"]
        prompt = st.session_state.get("resid_prompt", prompt)

    str_tokens = inspector.to_str_tokens(prompt)
    residual = inspector.get_residual_stream(cache)  # [n_layers+1, seq, d_model]
    n_positions = residual.shape[0]
    layer_labels = ["Embed"] + [f"L{i}" for i in range(inspector.n_layers)]

    token_idx = st.sidebar.selectbox(
        t("residual.select_token", lang),
        options=list(range(len(str_tokens))),
        format_func=lambda i: f"{i}: '{str_tokens[i]}'",
        key="resid_token",
    )

    # ── Norm per layer ──────────────────────────────────────────────────────
    norms = residual.norm(dim=-1).cpu().float().numpy()  # [n_layers+1, seq]

    fig_norm = go.Figure()
    for tok_i, tok in enumerate(str_tokens):
        opacity = 1.0 if tok_i == token_idx else 0.2
        width = 3 if tok_i == token_idx else 1
        fig_norm.add_trace(go.Scatter(
            x=layer_labels,
            y=norms[:, tok_i],
            mode="lines+markers",
            name=f"'{tok}'",
            opacity=opacity,
            line=dict(width=width),
        ))
    fig_norm.update_layout(
        title=t("residual.norm_title", lang),
        xaxis_title=t("residual.layer_label", lang),
        yaxis_title=t("residual.norm_label", lang),
        height=350,
        legend=dict(orientation="h", y=-0.2),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1E293B",
    )
    st.plotly_chart(fig_norm, use_container_width=True)

    # ── PCA projection ──────────────────────────────────────────────────────
    st.subheader(t("residual.pca_title", lang))

    resid_np = residual.cpu().float().numpy()  # [n_layers+1, seq, d_model]
    n_l, n_s, d = resid_np.shape
    flat = resid_np.reshape(-1, d)

    flat_c = flat - flat.mean(axis=0)
    _, _, Vt = np.linalg.svd(flat_c, full_matrices=False)
    pc = flat_c @ Vt[:2].T  # [n_l*n_s, 2]
    pc = pc.reshape(n_l, n_s, 2)

    fig_pca = go.Figure()
    for tok_i, tok in enumerate(str_tokens):
        opacity = 1.0 if tok_i == token_idx else 0.3
        size = 10 if tok_i == token_idx else 5
        fig_pca.add_trace(go.Scatter(
            x=pc[:, tok_i, 0],
            y=pc[:, tok_i, 1],
            mode="lines+markers+text",
            text=layer_labels,
            textposition="top center",
            name=f"'{tok}'",
            opacity=opacity,
            marker=dict(size=size),
        ))
    fig_pca.update_layout(
        xaxis_title="PC1",
        yaxis_title="PC2",
        height=450,
        legend=dict(orientation="h", y=-0.2),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1E293B",
    )
    st.plotly_chart(fig_pca, use_container_width=True)

else:
    st.info(t("error.no_cache", lang))
