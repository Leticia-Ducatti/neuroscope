"""
Fase 1B — detecção de Induction Heads (Olsson et al., 2022).

Protocolo:
  1. Gerar sequências repetidas do tipo [A1...An A1...An].
  2. Calcular prefix matching score: quanto cada head atende ao token t-1
     relativo ao match da primeira metade.
  3. Calcular copying score: quanto o output de cada head aumenta a
     probabilidade do token que ela estava atendendo.

Heads com ambos os scores altos são candidatas a induction heads.
"""

from __future__ import annotations

import random

import torch
import einops
from torch import Tensor

from src.inspector.model_inspector import ModelInspector


# ---------------------------------------------------------------------------
# Geração de sequências
# ---------------------------------------------------------------------------

def make_repeated_tokens(
    model: ModelInspector,
    seq_len: int = 50,
    n_repeats: int = 2,
) -> Tensor:
    """
    Gera uma sequência aleatória de tokens e a repete n_repeats vezes.
    Retorna tensor de shape [1, seq_len * n_repeats].

    Em induction heads, ao ver token A na segunda repetição, a head
    deve atender ao token imediatamente anterior ao A na primeira repetição.
    """
    vocab_size = model.model.cfg.d_vocab
    # Evita tokens especiais (BOS/EOS geralmente nos extremos do vocab)
    random_tokens = torch.randint(1000, vocab_size - 1000, (1, seq_len))
    repeated = random_tokens.repeat(1, n_repeats)
    return repeated.to(next(model.model.parameters()).device)


# ---------------------------------------------------------------------------
# Prefix Matching Score
# ---------------------------------------------------------------------------

def prefix_matching_score(
    inspector: ModelInspector,
    n_sequences: int = 50,
    seq_len: int = 50,
) -> Tensor:
    """
    Para cada head (layer, head), mede a atenção média que os tokens
    da segunda metade prestam ao token anterior ao seu par na primeira metade.

    Ex: sequência [A B C A B C]
        Posição 4 (segundo B) deve atender à posição 1 (A, antes do primeiro B).

    Retorna tensor [n_layers, n_heads] com scores entre 0 e 1.
    Scores altos indicam comportamento de induction head.
    """
    model = inspector.model
    n_layers = inspector.n_layers
    n_heads = inspector.n_heads

    scores = torch.zeros(n_layers, n_heads)

    for _ in range(n_sequences):
        tokens = make_repeated_tokens(inspector, seq_len=seq_len)  # [1, 2*seq_len]
        _, cache = inspector.run_tokens(tokens)

        for layer in range(n_layers):
            # pattern: [n_heads, seq_q, seq_k]
            pattern = inspector.get_attention_patterns(cache, layer)

            # Para cada posição na segunda metade (seq_len ... 2*seq_len-1),
            # a "fonte correta" é a posição correspondente na primeira metade - 1.
            # Ex: posição seq_len+i deve atender à posição i-1.
            for i in range(1, seq_len):
                q_pos = seq_len + i       # posição na segunda metade
                k_pos = i - 1             # posição correta na primeira metade
                # Atenção que cada head presta à posição correta
                scores[layer] += pattern[:, q_pos, k_pos]

    # Normaliza pelo número de sequências e posições consideradas
    scores /= n_sequences * (seq_len - 1)
    return scores


# ---------------------------------------------------------------------------
# Copying Score
# ---------------------------------------------------------------------------

def copying_score(
    inspector: ModelInspector,
    n_sequences: int = 50,
    seq_len: int = 50,
) -> Tensor:
    """
    Mede se o output de cada head aumenta a probabilidade do token
    para o qual ela está atendendo (comportamento de "cópia").

    Para cada head, injeta apenas o output dessa head no residual stream
    e mede o logit do token atendido vs. tokens aleatórios.

    Retorna tensor [n_layers, n_heads].
    Scores positivos indicam que a head copia o token atendido.
    """
    model = inspector.model
    n_layers = inspector.n_layers
    n_heads = inspector.n_heads

    scores = torch.zeros(n_layers, n_heads)

    for _ in range(n_sequences):
        tokens = make_repeated_tokens(inspector, seq_len=seq_len)  # [1, 2*seq_len]
        _, cache = inspector.run_tokens(tokens)

        for layer in range(n_layers):
            pattern = inspector.get_attention_patterns(cache, layer)  # [n_heads, seq, seq]

            # hook_z: output por head antes de W_O — shape [batch, seq, n_heads, d_head]
            attn_result = cache["z", layer].squeeze(0)  # [seq, n_heads, d_head]

            # W_O projeta de [n_heads * d_head] para [d_model]
            W_O = model.blocks[layer].attn.W_O  # [n_heads, d_head, d_model]

            for head in range(n_heads):
                head_out = attn_result[:, head, :]  # [seq, d_head]
                # Projeta via W_O para o espaço do residual
                head_contrib = head_out @ W_O[head]  # [seq, d_model]

                # Para posições na segunda metade, vê qual token a head mais atende
                head_scores_list = []
                for i in range(1, seq_len):
                    q_pos = seq_len + i
                    # Token mais atendido por esta head nesta posição
                    attended_pos = pattern[head, q_pos].argmax().item()
                    attended_token = tokens[0, attended_pos].item()

                    # Logit do token atendido via contribuição desta head
                    contrib_logit = model.unembed(
                        model.ln_final(head_contrib[q_pos].unsqueeze(0).unsqueeze(0))
                    ).squeeze()

                    logit_attended = contrib_logit[attended_token].item()
                    # Score normalizado: logit do token atendido vs. média
                    logit_mean = contrib_logit.mean().item()
                    head_scores_list.append(logit_attended - logit_mean)

                scores[layer, head] += sum(head_scores_list) / len(head_scores_list)

    scores /= n_sequences
    return scores


# ---------------------------------------------------------------------------
# Score combinado e ranking
# ---------------------------------------------------------------------------

def induction_scores(
    inspector: ModelInspector,
    n_sequences: int = 50,
    seq_len: int = 50,
) -> dict[str, Tensor]:
    """
    Calcula prefix matching score e copying score e retorna ambos.
    """
    pm = prefix_matching_score(inspector, n_sequences, seq_len)
    cp = copying_score(inspector, n_sequences, seq_len)
    return {"prefix_matching": pm, "copying": cp}


def top_induction_heads(
    scores: dict[str, Tensor],
    n_top: int = 10,
    mode: str = "prefix_matching",
) -> list[tuple[int, int, float]]:
    """
    Retorna as top-N heads ordenadas pelo score especificado.
    Cada item: (layer, head, score).
    """
    s = scores[mode]
    flat = s.flatten()
    top_vals, top_idx = flat.topk(n_top)
    n_heads = s.shape[1]
    result = []
    for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
        layer = idx // n_heads
        head = idx % n_heads
        result.append((layer, head, val))
    return result
