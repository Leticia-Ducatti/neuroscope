"""
Fase 1C — IOI Circuit (Wang et al., 2022).

Protocolo:
  1. Medir acurácia base do GPT-2 small na tarefa IOI.
  2. Usar activation patching para isolar a contribuição causal
     de cada atenção head na logit difference IO - S.
  3. Identificar as Name Mover Heads (aquelas com maior contribuição positiva).

Referência: Wang et al. (2022). Interpretability in the Wild:
  a Circuit for Indirect Object Identification in GPT-2 small.
"""

from __future__ import annotations

import json
from pathlib import Path
from functools import partial

import torch
from torch import Tensor

from src.inspector.model_inspector import ModelInspector


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def load_ioi_prompts(path: str | Path | None = None) -> list[dict]:
    if path is None:
        path = Path(__file__).parent.parent.parent / "data" / "prompts" / "ioi_prompts.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def corrupt_prompt(prompt_data: dict) -> str:
    """
    Gera versão corrompida do prompt trocando sujeito e objeto indireto.
    Ex: 'John gave ... to Mary' → 'Mary gave ... to John'
    """
    p = prompt_data["prompt"]
    s = prompt_data["subject"]
    io = prompt_data["indirect_object"]
    # Troca simples: IO → placeholder, S → IO, placeholder → S
    corrupted = p.replace(io, "__IO__").replace(s, io).replace("__IO__", s)
    return corrupted


# ---------------------------------------------------------------------------
# Baseline de acurácia
# ---------------------------------------------------------------------------

def logit_diff(
    inspector: ModelInspector,
    prompt: str,
    correct_token: str,
    incorrect_token: str,
) -> float:
    """
    Retorna logit(IO) - logit(S) na última posição.
    Positivo = modelo prefere a resposta correta.
    """
    model = inspector.model
    logits, _ = inspector.run(prompt)
    last_logits = logits[0, -1, :]  # [vocab_size]

    io_id = model.to_single_token(correct_token)
    s_id = model.to_single_token(incorrect_token)

    return (last_logits[io_id] - last_logits[s_id]).item()


def ioi_accuracy(inspector: ModelInspector, prompts: list[dict]) -> dict:
    """
    Fração de prompts onde logit(IO) > logit(S) e logit diff médio.
    """
    diffs = []
    for p in prompts:
        d = logit_diff(inspector, p["prompt"], p["correct_token"], p["incorrect_token"])
        diffs.append(d)

    correct = sum(1 for d in diffs if d > 0)
    return {
        "accuracy": correct / len(diffs),
        "mean_logit_diff": sum(diffs) / len(diffs),
        "logit_diffs": diffs,
    }


# ---------------------------------------------------------------------------
# Activation Patching
# ---------------------------------------------------------------------------

def _patch_hook(value: Tensor, hook, patch_value: Tensor):
    """Hook que substitui o tensor inteiro pelo valor do prompt corrompido."""
    return patch_value


def patch_attention_head_output(
    inspector: ModelInspector,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_token: str,
    incorrect_token: str,
    layer: int,
    head: int,
) -> float:
    """
    Substitui o output da head (layer, head) pela versão do prompt corrompido.
    Retorna a logit diff resultante.

    Uma logit diff alta (próxima do baseline) indica que esta head não é crítica.
    Uma logit diff baixa (próxima de zero ou negativa) indica que esta head é
    importante para a resposta correta.
    """
    model = inspector.model

    # Cache do prompt corrompido
    corrupted_tokens = model.to_tokens(corrupted_prompt)
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    # hook_z: output por head antes de W_O — shape [batch, seq, n_heads, d_head]
    patch_value = corrupted_cache["z", layer][:, :, head:head+1, :]

    hook_name = f"blocks.{layer}.attn.hook_z"

    def hook_fn(value: Tensor, hook):
        # Substitui apenas a head especificada
        value[:, :, head:head+1, :] = patch_value[:, :value.shape[1], :]
        return value

    # Forward pass do prompt limpo com a head patcheada
    clean_tokens = model.to_tokens(clean_prompt)
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    last_logits = patched_logits[0, -1, :]
    io_id = model.to_single_token(correct_token)
    s_id = model.to_single_token(incorrect_token)
    return (last_logits[io_id] - last_logits[s_id]).item()


def patch_mlp_output(
    inspector: ModelInspector,
    clean_prompt: str,
    corrupted_prompt: str,
    correct_token: str,
    incorrect_token: str,
    layer: int,
) -> float:
    """
    Substitui o output completo do MLP na camada especificada.
    Retorna a logit diff resultante.
    """
    model = inspector.model

    corrupted_tokens = model.to_tokens(corrupted_prompt)
    with torch.no_grad():
        _, corrupted_cache = model.run_with_cache(corrupted_tokens)

    patch_value = corrupted_cache["mlp_out", layer]
    hook_name = f"blocks.{layer}.hook_mlp_out"

    def hook_fn(value: Tensor, hook):
        seq_len = value.shape[1]
        value[:, :seq_len, :] = patch_value[:, :seq_len, :]
        return value

    clean_tokens = model.to_tokens(clean_prompt)
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            clean_tokens,
            fwd_hooks=[(hook_name, hook_fn)],
        )

    last_logits = patched_logits[0, -1, :]
    io_id = model.to_single_token(correct_token)
    s_id = model.to_single_token(incorrect_token)
    return (last_logits[io_id] - last_logits[s_id]).item()


# ---------------------------------------------------------------------------
# Mapa completo de contribuições
# ---------------------------------------------------------------------------

def compute_patching_map(
    inspector: ModelInspector,
    prompts: list[dict],
) -> dict[str, Tensor]:
    """
    Executa activation patching em todas as heads e MLPs para todos os prompts.

    Retorna:
      {
        "attn": Tensor [n_layers, n_heads],  — logit diff médio pós-patching
        "mlp":  Tensor [n_layers],            — logit diff médio pós-patching
        "baseline": float,                    — logit diff sem patching
        "baseline_corrupted": float,          — logit diff do prompt corrompido
      }

    Interpretação:
      - Valor próximo do baseline → componente não crítico.
      - Valor próximo de zero (ou negativo) → componente crucial para a tarefa.
      - Normaliza-se como: (patched - corrupted) / (baseline - corrupted)
        → 1.0 = componente recupera toda a performance, 0.0 = nada.
    """
    n_layers = inspector.n_layers
    n_heads = inspector.n_heads

    attn_map = torch.zeros(n_layers, n_heads)
    mlp_map = torch.zeros(n_layers)
    baselines = []
    baselines_corrupted = []

    for p in prompts:
        clean = p["prompt"]
        corrupted = corrupt_prompt(p)
        io_tok = p["correct_token"]
        s_tok = p["incorrect_token"]

        baselines.append(logit_diff(inspector, clean, io_tok, s_tok))
        baselines_corrupted.append(logit_diff(inspector, corrupted, io_tok, s_tok))

        for layer in range(n_layers):
            # MLP
            mlp_map[layer] += patch_mlp_output(
                inspector, clean, corrupted, io_tok, s_tok, layer
            )
            # Attention heads
            for head in range(n_heads):
                attn_map[layer, head] += patch_attention_head_output(
                    inspector, clean, corrupted, io_tok, s_tok, layer, head
                )

    n = len(prompts)
    attn_map /= n
    mlp_map /= n
    baseline = sum(baselines) / n
    baseline_corrupted = sum(baselines_corrupted) / n

    # Normaliza para [0, 1]: quanto cada componente "recupera" a performance
    denom = baseline - baseline_corrupted
    if abs(denom) > 1e-6:
        attn_map_norm = (attn_map - baseline_corrupted) / denom
        mlp_map_norm = (mlp_map - baseline_corrupted) / denom
    else:
        attn_map_norm = attn_map
        mlp_map_norm = mlp_map

    return {
        "attn": attn_map_norm,
        "mlp": mlp_map_norm,
        "attn_raw": attn_map,
        "mlp_raw": mlp_map,
        "baseline": baseline,
        "baseline_corrupted": baseline_corrupted,
    }


# ---------------------------------------------------------------------------
# Ranking de heads
# ---------------------------------------------------------------------------

def top_name_mover_heads(
    patching_map: dict[str, Tensor],
    n_top: int = 10,
) -> list[tuple[int, int, float]]:
    """
    Retorna as heads com maior contribuição positiva normalizada (Name Mover Heads).
    Cada item: (layer, head, normalized_score).
    """
    attn = patching_map["attn"]
    flat = attn.flatten()
    top_vals, top_idx = flat.topk(n_top)
    n_heads = attn.shape[1]
    return [
        (int(idx // n_heads), int(idx % n_heads), float(val))
        for val, idx in zip(top_vals.tolist(), top_idx.tolist())
    ]
