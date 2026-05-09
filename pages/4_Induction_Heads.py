import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import pandas as pd
import torch

from src.viz.session import inject_global_css, sidebar_language, sidebar_model, page_header, require_model
from src.viz.i18n import t
from src.circuits.induction_utils import induction_scores, top_induction_heads

st.set_page_config(page_title="Induction Heads — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
inspector = sidebar_model(lang)

page_header(t("induction.title", lang), t("induction.intro", lang))

if not require_model(inspector, lang):
    st.stop()

n_sequences = st.sidebar.slider(t("sidebar.n_sequences", lang), 10, 100, 30, step=10, key="ind_nseq")
seq_len = st.sidebar.slider(t("sidebar.seq_len", lang), 20, 100, 50, step=10, key="ind_seqlen")
run_btn = st.sidebar.button(t("induction.run_button", lang), type="primary")

PAPER_HEADS = {(1, 0), (1, 4)}

if run_btn or "ind_scores" in st.session_state:
    if run_btn:
        with st.spinner(t("induction.computing", lang, n=n_sequences, l=seq_len)):
            scores = induction_scores(inspector, n_sequences=n_sequences, seq_len=seq_len)
            st.session_state["ind_scores"] = scores
    else:
        scores = st.session_state["ind_scores"]

    n_layers = inspector.n_layers
    n_heads = inspector.n_heads
    layer_labels = [f"L{l}" for l in range(n_layers)]
    head_labels = [f"H{h}" for h in range(n_heads)]

    col1, col2 = st.columns(2)

    _heatmap_layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    # ── Prefix Matching Score ──────────────────────────────────────────────
    with col1:
        pm = scores["prefix_matching"].cpu().float().numpy()
        fig_pm = px.imshow(
            pm,
            x=head_labels,
            y=layer_labels,
            title=t("induction.pm_title", lang),
            labels={"x": "Head", "y": "Layer", "color": "Score"},
            color_continuous_scale="RdBu",
            aspect="auto",
            template="plotly_dark",
        )
        for (l, h) in PAPER_HEADS:
            fig_pm.add_annotation(x=h, y=l, text="★", showarrow=False,
                                  font=dict(size=14, color="white"))
        fig_pm.update_layout(**_heatmap_layout)
        st.plotly_chart(fig_pm, use_container_width=True)

    # ── Copying Score ──────────────────────────────────────────────────────
    with col2:
        cp = scores["copying"].cpu().float().numpy()
        fig_cp = px.imshow(
            cp,
            x=head_labels,
            y=layer_labels,
            title=t("induction.cp_title", lang),
            labels={"x": "Head", "y": "Layer", "color": "Score"},
            color_continuous_scale="RdBu",
            aspect="auto",
            template="plotly_dark",
        )
        for (l, h) in PAPER_HEADS:
            fig_cp.add_annotation(x=h, y=l, text="★", showarrow=False,
                                  font=dict(size=14, color="white"))
        fig_cp.update_layout(**_heatmap_layout)
        st.plotly_chart(fig_cp, use_container_width=True)

    st.caption(t("induction.paper_note", lang))

    # ── Top heads table ────────────────────────────────────────────────────
    st.subheader(t("induction.top_heads", lang))
    top = top_induction_heads(scores, n_top=10, mode="prefix_matching")
    rows = []
    for layer, head, score in top:
        head_id = f"L{layer}H{head}"
        is_paper = "★" if (layer, head) in PAPER_HEADS else ""
        rows.append({t("induction.col_head", lang): f"{is_paper} {head_id}".strip(),
                     t("induction.col_score", lang): round(score, 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.info(t("error.no_cache", lang))
    st.write(t("induction.intro", lang))
