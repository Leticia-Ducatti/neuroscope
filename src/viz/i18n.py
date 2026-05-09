"""
Internationalization module — English (default) and Portuguese (pt-BR).
Usage:
    from src.viz.i18n import t, LANGUAGES
    label = t("attention_explorer.title")
"""

from __future__ import annotations

LANGUAGES = {"English": "en", "Português (BR)": "pt"}

_STRINGS: dict[str, dict[str, str]] = {
    # -----------------------------------------------------------------------
    # Shared / sidebar
    # -----------------------------------------------------------------------
    "sidebar.language": {"en": "Language", "pt": "Idioma"},
    "sidebar.model": {"en": "Model", "pt": "Modelo"},
    "sidebar.model_loading": {"en": "Loading model…", "pt": "Carregando modelo…"},
    "sidebar.model_loaded": {"en": "Model loaded", "pt": "Modelo carregado"},
    "sidebar.prompt": {"en": "Input prompt", "pt": "Prompt de entrada"},
    "sidebar.run": {"en": "Run", "pt": "Executar"},
    "sidebar.running": {"en": "Running…", "pt": "Executando…"},
    "sidebar.layer": {"en": "Layer", "pt": "Camada"},
    "sidebar.head": {"en": "Head", "pt": "Head"},
    "sidebar.top_k": {"en": "Top-K tokens", "pt": "Top-K tokens"},
    "sidebar.n_sequences": {"en": "Sequences (scoring)", "pt": "Sequências (scoring)"},
    "sidebar.seq_len": {"en": "Sequence length", "pt": "Comprimento da sequência"},
    "sidebar.n_prompts": {"en": "Prompts (patching)", "pt": "Prompts (patching)"},

    # -----------------------------------------------------------------------
    # Home
    # -----------------------------------------------------------------------
    "home.title": {"en": "NeuroScope", "pt": "NeuroScope"},
    "home.subtitle": {
        "en": "Mechanistic interpretability for transformer language models.",
        "pt": "Interpretabilidade mecanística para modelos de linguagem transformer.",
    },
    "home.description": {
        "en": (
            "NeuroScope lets you inspect the internal computations of GPT-2 small, "
            "reproducing results from Olsson et al. (2022) and Wang et al. (2022). "
            "Select a page in the sidebar to begin."
        ),
        "pt": (
            "O NeuroScope permite inspecionar as computações internas do GPT-2 small, "
            "reproduzindo resultados de Olsson et al. (2022) e Wang et al. (2022). "
            "Selecione uma página na barra lateral para começar."
        ),
    },
    "home.pages_title": {"en": "Available pages", "pt": "Páginas disponíveis"},
    "home.page_attention": {
        "en": "**Attention Explorer** — Visualise attention patterns per layer and head.",
        "pt": "**Attention Explorer** — Visualize padrões de atenção por camada e head.",
    },
    "home.page_residual": {
        "en": "**Residual Stream** — Track how the residual stream evolves across layers.",
        "pt": "**Residual Stream** — Acompanhe a evolução do residual stream pelas camadas.",
    },
    "home.page_logit_lens": {
        "en": "**Logit Lens** — See token predictions layer by layer.",
        "pt": "**Logit Lens** — Veja as previsões de tokens camada por camada.",
    },
    "home.page_induction": {
        "en": "**Induction Heads** — Detect induction heads via prefix matching score.",
        "pt": "**Induction Heads** — Detecte induction heads via prefix matching score.",
    },
    "home.page_ioi": {
        "en": "**IOI Circuit** — Map attention head contributions via activation patching.",
        "pt": "**IOI Circuit** — Mapeie contribuições de heads via activation patching.",
    },

    # -----------------------------------------------------------------------
    # Attention Explorer
    # -----------------------------------------------------------------------
    "attention.title": {"en": "Attention Explorer", "pt": "Attention Explorer"},
    "attention.intro": {
        "en": "Visualise softmaxed attention weights for each layer and head.",
        "pt": "Visualize os pesos de atenção (softmax) para cada camada e head.",
    },
    "attention.select_heads": {"en": "Heads to display", "pt": "Heads a exibir"},
    "attention.all_heads": {"en": "All heads", "pt": "Todas as heads"},
    "attention.heatmap_title": {
        "en": "Attention pattern — Layer {layer} Head {head}",
        "pt": "Padrão de atenção — Camada {layer} Head {head}",
    },
    "attention.axis_key": {"en": "Key (attended token)", "pt": "Key (token atendido)"},
    "attention.axis_query": {"en": "Query (current token)", "pt": "Query (token atual)"},

    # -----------------------------------------------------------------------
    # Residual Stream
    # -----------------------------------------------------------------------
    "residual.title": {"en": "Residual Stream", "pt": "Residual Stream"},
    "residual.intro": {
        "en": (
            "The residual stream carries information across layers. "
            "Each layer reads from it and writes back to it. "
            "Below: PCA projection of the residual stream states across layers."
        ),
        "pt": (
            "O residual stream carrega informação pelas camadas. "
            "Cada camada lê dele e escreve nele. "
            "Abaixo: projeção PCA dos estados do residual stream pelas camadas."
        ),
    },
    "residual.select_token": {"en": "Token position to highlight", "pt": "Posição do token a destacar"},
    "residual.pca_title": {
        "en": "Residual stream — PCA projection (PC1 × PC2)",
        "pt": "Residual stream — Projeção PCA (PC1 × PC2)",
    },
    "residual.norm_title": {
        "en": "Residual stream norm per layer",
        "pt": "Norma do residual stream por camada",
    },
    "residual.layer_label": {"en": "Layer", "pt": "Camada"},
    "residual.norm_label": {"en": "L2 Norm", "pt": "Norma L2"},

    # -----------------------------------------------------------------------
    # Logit Lens
    # -----------------------------------------------------------------------
    "logit_lens.title": {"en": "Logit Lens", "pt": "Logit Lens"},
    "logit_lens.intro": {
        "en": (
            "Projects intermediate residual stream states into token space to reveal "
            "what the model 'thinks' at each layer before the final prediction."
        ),
        "pt": (
            "Projeta estados intermediários do residual stream no espaço de tokens "
            "para revelar o que o modelo 'pensa' em cada camada antes da predição final."
        ),
    },
    "logit_lens.table_title": {
        "en": "Top-{k} predictions per layer (last token position)",
        "pt": "Top-{k} previsões por camada (última posição de token)",
    },
    "logit_lens.col_layer": {"en": "Layer", "pt": "Camada"},
    "logit_lens.col_rank": {"en": "Rank", "pt": "Rank"},
    "logit_lens.col_token": {"en": "Token", "pt": "Token"},
    "logit_lens.col_prob": {"en": "Probability", "pt": "Probabilidade"},
    "logit_lens.convergence": {
        "en": "Layer where top-1 token first matches final prediction",
        "pt": "Camada onde o token top-1 coincide pela primeira vez com a predição final",
    },

    # -----------------------------------------------------------------------
    # Induction Heads
    # -----------------------------------------------------------------------
    "induction.title": {"en": "Induction Heads", "pt": "Induction Heads"},
    "induction.intro": {
        "en": (
            "Detects induction heads using repeated random token sequences. "
            "An induction head attends to the token preceding the current token's "
            "previous occurrence — enabling in-context copying."
        ),
        "pt": (
            "Detecta induction heads usando sequências repetidas de tokens aleatórios. "
            "Uma induction head atende ao token anterior à ocorrência anterior do token atual — "
            "habilitando cópia in-context."
        ),
    },
    "induction.run_button": {"en": "Compute scores", "pt": "Calcular scores"},
    "induction.computing": {
        "en": "Computing scores ({n} sequences × {l} seq_len)… this may take a few minutes.",
        "pt": "Calculando scores ({n} sequências × {l} seq_len)… pode levar alguns minutos.",
    },
    "induction.pm_title": {"en": "Prefix Matching Score", "pt": "Prefix Matching Score"},
    "induction.cp_title": {"en": "Copying Score", "pt": "Copying Score"},
    "induction.top_heads": {"en": "Top induction heads (prefix matching)", "pt": "Top induction heads (prefix matching)"},
    "induction.col_head": {"en": "Head", "pt": "Head"},
    "induction.col_score": {"en": "Score", "pt": "Score"},
    "induction.paper_note": {
        "en": "★ marks heads reported in Olsson et al. (2022) for GPT-2 small: L1H0, L1H4.",
        "pt": "★ marca heads reportadas em Olsson et al. (2022) para GPT-2 small: L1H0, L1H4.",
    },

    # -----------------------------------------------------------------------
    # IOI Circuit
    # -----------------------------------------------------------------------
    "ioi.title": {"en": "IOI Circuit", "pt": "IOI Circuit"},
    "ioi.intro": {
        "en": (
            "Maps the causal contribution of each attention head to the "
            "Indirect Object Identification task via activation patching. "
            "A normalised score near 1.0 means the head is critical for the correct answer."
        ),
        "pt": (
            "Mapeia a contribuição causal de cada head de atenção para a tarefa de "
            "Indirect Object Identification via activation patching. "
            "Score normalizado próximo de 1.0 indica head crítica para a resposta correta."
        ),
    },
    "ioi.run_button": {"en": "Run patching", "pt": "Executar patching"},
    "ioi.computing": {
        "en": "Running activation patching ({n} prompts)… may take 5-15 min.",
        "pt": "Executando activation patching ({n} prompts)… pode levar 5-15 min.",
    },
    "ioi.baseline": {"en": "Baseline logit diff (clean)", "pt": "Logit diff baseline (clean)"},
    "ioi.baseline_corrupted": {
        "en": "Baseline logit diff (corrupted)",
        "pt": "Logit diff baseline (corrompido)",
    },
    "ioi.accuracy": {"en": "IOI accuracy", "pt": "Acurácia IOI"},
    "ioi.heatmap_title": {
        "en": "Activation patching — attention head contributions (normalised)",
        "pt": "Activation patching — contribuição das attention heads (normalizada)",
    },
    "ioi.mlp_title": {"en": "MLP contributions per layer", "pt": "Contribuição dos MLPs por camada"},
    "ioi.top_heads": {"en": "Top Name Mover Heads", "pt": "Top Name Mover Heads"},
    "ioi.paper_note": {
        "en": "★ marks Name Mover Heads from Wang et al. (2022): L9H6, L9H9, L10H0.",
        "pt": "★ marca Name Mover Heads de Wang et al. (2022): L9H6, L9H9, L10H0.",
    },

    # -----------------------------------------------------------------------
    # Errors / misc
    # -----------------------------------------------------------------------
    "error.no_cache": {
        "en": "Run a prompt first to see results.",
        "pt": "Execute um prompt primeiro para ver os resultados.",
    },
    "error.model_not_loaded": {
        "en": "Model not loaded. Check the sidebar.",
        "pt": "Modelo não carregado. Verifique a barra lateral.",
    },
}


def t(key: str, lang: str = "en", **kwargs) -> str:
    """
    Translate key to the given language.
    Falls back to English if key or lang is missing.
    Supports format kwargs: t("foo.bar", lang, layer=3, head=1)
    """
    entry = _STRINGS.get(key, {})
    text = entry.get(lang) or entry.get("en") or key
    if kwargs:
        text = text.format(**kwargs)
    return text
