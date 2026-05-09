import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
from src.viz.session import inject_global_css, sidebar_language, sidebar_model

st.set_page_config(page_title="Guide — NeuroScope", layout="wide")

inject_global_css()
lang = sidebar_language()
sidebar_model(lang)

st.markdown("""
<div class="ns-banner">
    <div class="ns-banner-title">📖 Guide</div>
    <div class="ns-banner-desc">How to use NeuroScope — explained for everyone, no ML background needed.</div>
</div>
""", unsafe_allow_html=True)

# ── Before you start ──────────────────────────────────────────────────────────
st.markdown("""
<div class="ns-card" style="margin-bottom:1.5rem;">
    <span class="ns-card-icon">🚀</span>
    <div>
        <div class="ns-card-title" style="font-size:1rem; margin-bottom:0.5rem;">Before you start</div>
        <div class="ns-card-desc" style="font-size:0.88rem;">
            Use the <strong style="color:#E2E8F0;">sidebar on the left</strong> to load a model — start with <code>gpt2</code>, it's the fastest.<br>
            Click <strong style="color:#818CF8;">⚡ Load</strong> and wait for the spinner to finish before navigating to any page.<br>
            Once loaded, type your own prompt in <strong style="color:#E2E8F0;">Input prompt</strong> or use the default one.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Pages ─────────────────────────────────────────────────────────────────────
PAGES = [
    {
        "icon": "🧠",
        "title": "Attention Explorer — Where is the AI looking?",
        "what": "When reading a sentence, the AI doesn't pay equal attention to all words. It 'focuses' on certain words more than others to decide what comes next — this mechanism is called **attention**.",
        "see": "A grid of coloured squares (heatmap). Each **row** is a word the AI is **currently processing**, and each **column** is a word it is **consulting**. The darker the square, the more attention is being paid.",
        "interpret": [
            "Dark square at (row 'Mary', column 'John') → *while processing 'Mary', the AI is heavily looking at 'John'*",
            "Dark squares along the diagonal → a word is attending to itself",
            "Dark squares on the last row → the AI is deciding the next word based on those words",
        ],
        "controls": [
            ("Layer", "Which processing step to inspect (0 = start, 11 = end). Early layers capture grammar, later layers capture meaning."),
            ("Heads", "Each layer has several independent 'focus points'. Select which ones to display."),
        ],
    },
    {
        "icon": "📊",
        "title": "Residual Stream — How does the reasoning evolve?",
        "what": "The AI doesn't process text all at once. It passes through 12 layers in sequence, **adding information** at each step — like a draft being progressively refined.",
        "see": "**Norm chart:** shows the 'size' of each word's internal representation at each layer. A larger value means the AI is carrying more information about that word.\n\n**PCA chart (2D map):** projects the internal representations into 2D so you can see how they move through the layers. Each dot is a layer, lines show the path.",
        "interpret": [
            "A word whose curve **grows a lot** in the last layers is gaining importance for the final prediction",
            "In the PCA chart, words with **longer trajectories** are being transformed more heavily",
            "Words with similar trajectories tend to play similar roles in the sentence (e.g. two proper nouns)",
        ],
        "controls": [
            ("Token to highlight", "Highlights one specific word in the charts to make it easier to follow."),
        ],
    },
    {
        "icon": "🔭",
        "title": "Logit Lens — What does the AI think comes next, at each step?",
        "what": "Imagine asking someone *'what word are you about to say?'* mid-thought, before they finish thinking. The Logit Lens does exactly that: it 'pauses' the AI at each layer and asks what the next word would be at that moment.",
        "see": "A bar chart showing, for each layer (Embed, L0, L1 … L11), which token the AI would most likely choose and with what confidence. A message below tells you **at which layer the AI first converged** to its final answer.",
        "interpret": [
            "Low bars in early layers → the AI doesn't know the answer yet (normal)",
            "High bars in the last layers → high confidence in the prediction",
            "Early convergence (e.g. layer 4 of 12) → the AI recognised the pattern quickly",
            "Late or no convergence → the sentence is ambiguous or hard for the model",
        ],
        "controls": [
            ("Top-K tokens", "How many candidate tokens to show per layer in the full table."),
        ],
    },
    {
        "icon": "🔁",
        "title": "Induction Heads — Can the AI copy patterns it has seen before?",
        "what": "A fundamental ability of language AIs is recognising patterns like *'last time I saw X, Y followed'*. The parts of the model responsible for this are called **induction heads**.\n\nExample: in the text `A B C A B`, upon seeing the second `A`, an induction head detects *'I've seen A before, and B followed'* and votes for `B` as the next word.",
        "see": "Two heatmaps — one for each test:\n\n**Prefix Matching Score:** measures whether the head is looking at the token just before its previous occurrence.\n\n**Copying Score:** measures whether the head's output increases the probability of the token it was attending to.\n\nThe ★ symbol marks the heads identified in the original paper (Olsson et al., 2022).",
        "interpret": [
            "**Dark red squares** → high score → candidate induction head",
            "If the ★ marks are in the reddest regions, results match the published research",
            "Induction heads in GPT-2 typically appear at layer 1 (L1H0 and L1H4)",
        ],
        "controls": [
            ("Sequences", "How many random sequences to use. More = more accurate, but slower."),
            ("Seq len", "Length of each test sequence."),
        ],
    },
    {
        "icon": "🔗",
        "title": "IOI Circuit — How does the AI know who receives the object?",
        "what": 'Consider: *"When Mary and John went to the store, John gave a bottle of milk to ___"*\n\nThe correct answer is **Mary** (not John, who already appeared as the giver). This task — identifying the correct indirect object — is called **IOI (Indirect Object Identification)**.\n\nThe experiment uses **activation patching**: it processes the original sentence and a corrupted version (with names swapped), then injects parts of the corrupted processing into the original to measure what each component contributes.',
        "see": "**Three metrics at the top:**\n- *IOI accuracy* — % of sentences where the AI picks the right name\n- *Baseline logit diff* — how much more confident the AI is in the correct vs. wrong answer\n- *Baseline corrupted* — same metric with swapped names (should be negative)\n\n**Attention heatmap:** each head's contribution to the correct answer. Score near **1.0** = critical head.\n\n**MLP bar chart:** same for the memory blocks of each layer.",
        "interpret": [
            "★ marks the **Name Mover Heads** from the paper (L9H6, L9H9, L10H0) — the most important for this task",
            "If those heads show the highest scores, results align with Wang et al. (2022)",
            "**Negative scores** = the head is actively hurting the correct answer (Negative Movers)",
            "Layers 9-11 tend to be the most important for this task",
        ],
        "controls": [
            ("Prompts", "How many IOI sentence examples to use. More = more reliable, but much slower — each prompt runs the model dozens of times."),
        ],
    },
]

for page in PAGES:
    with st.expander(f"{page['icon']}  {page['title']}", expanded=False):
        col_what, col_see = st.columns(2)

        with col_what:
            st.markdown("##### What is it?")
            st.markdown(page["what"])
            st.markdown("##### How to interpret")
            for tip in page["interpret"]:
                st.markdown(f"- {tip}")

        with col_see:
            st.markdown("##### What you see")
            st.markdown(page["see"])
            if page["controls"]:
                st.markdown("##### Controls")
                for ctrl, desc in page["controls"]:
                    st.markdown(f"**{ctrl}** — {desc}")

st.divider()
st.markdown("""
<p style='color:#64748B; font-size:0.82rem;'>
💡 <strong style='color:#94A3B8;'>Tips:</strong>
Load the model once and navigate freely between pages — it stays in memory.
Always run a prompt before adjusting sliders.
GPT-2 was trained in English — use English prompts for best results.
The IOI page is the slowest — start with 5-10 prompts.
</p>
""", unsafe_allow_html=True)
