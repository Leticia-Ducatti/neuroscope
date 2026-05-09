import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.viz.session import inject_global_css, sidebar_language, sidebar_model, page_header, require_model
from src.viz.i18n import t
from src.circuits.ioi_utils import (
    load_ioi_prompts,
    ioi_accuracy,
    compute_patching_map,
    top_name_mover_heads,
)

st.set_page_config(page_title="IOI Circuit — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
inspector = sidebar_model(lang)

page_header(t("ioi.title", lang), t("ioi.intro", lang))

if not require_model(inspector, lang):
    st.stop()

prompts = load_ioi_prompts()
n_prompts = st.sidebar.slider(
    t("sidebar.n_prompts", lang), 5, len(prompts), 10, step=5, key="ioi_n"
)
run_btn = st.sidebar.button(t("ioi.run_button", lang), type="primary")

PAPER_HEADS = {(9, 6), (9, 9), (10, 0)}

if run_btn or "ioi_map" in st.session_state:
    if run_btn:
        subset = prompts[:n_prompts]
        with st.spinner(t("ioi.computing", lang, n=n_prompts)):
            accuracy_result = ioi_accuracy(inspector, subset)
            patching_map = compute_patching_map(inspector, subset)
            st.session_state["ioi_map"] = patching_map
            st.session_state["ioi_acc"] = accuracy_result
    else:
        patching_map = st.session_state["ioi_map"]
        accuracy_result = st.session_state["ioi_acc"]

    # ── Metrics row ────────────────────────────────────────────────────────
    m1, m2, m3 = st.columns(3)
    m1.metric(t("ioi.accuracy", lang), f"{accuracy_result['accuracy']:.1%}")
    m2.metric(t("ioi.baseline", lang), f"{patching_map['baseline']:.3f}")
    m3.metric(t("ioi.baseline_corrupted", lang), f"{patching_map['baseline_corrupted']:.3f}")

    st.divider()

    n_layers = inspector.n_layers
    n_heads = inspector.n_heads
    layer_labels = [f"L{l}" for l in range(n_layers)]
    head_labels = [f"H{h}" for h in range(n_heads)]

    # ── Attention patching heatmap ─────────────────────────────────────────
    attn = patching_map["attn"].cpu().float().numpy()
    fig_attn = px.imshow(
        attn,
        x=head_labels,
        y=layer_labels,
        title=t("ioi.heatmap_title", lang),
        labels={"x": "Head", "y": "Layer", "color": "Score norm."},
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        aspect="auto",
        template="plotly_dark",
    )
    for (l, h) in PAPER_HEADS:
        fig_attn.add_annotation(x=h, y=l, text="★", showarrow=False,
                                font=dict(size=14, color="white"))
    fig_attn.update_layout(paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_attn, use_container_width=True)
    st.caption(t("ioi.paper_note", lang))

    # ── MLP contributions ──────────────────────────────────────────────────
    mlp = patching_map["mlp"].cpu().float().numpy()
    fig_mlp = go.Figure(go.Bar(
        x=layer_labels,
        y=mlp,
        marker_color=["#818CF8" if v > 0.1 else "#334155" for v in mlp],
    ))
    fig_mlp.update_layout(
        title=t("ioi.mlp_title", lang),
        xaxis_title=t("residual.layer_label", lang),
        yaxis_title="Score norm.",
        height=300,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#1E293B",
    )
    st.plotly_chart(fig_mlp, use_container_width=True)

    # ── Top heads table ────────────────────────────────────────────────────
    st.subheader(t("ioi.top_heads", lang))
    top = top_name_mover_heads(patching_map, n_top=10)
    rows = []
    for layer, head, score in top:
        head_id = f"L{layer}H{head}"
        is_paper = "★" if (layer, head) in PAPER_HEADS else ""
        rows.append({"Head": f"{is_paper} {head_id}".strip(), "Score norm.": round(score, 4)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

else:
    st.info(t("error.no_cache", lang))
    st.write(t("ioi.intro", lang))
