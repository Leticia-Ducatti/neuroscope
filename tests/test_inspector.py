"""
Fase 1A — testes unitários do ModelInspector.
Rodar com: pytest tests/test_inspector.py -v
O modelo GPT-2 small é baixado uma vez e reutilizado via session fixture.
"""

import pytest
import torch

from src.inspector.model_inspector import InspectorConfig, ModelInspector

PROMPT = "The quick brown fox jumps over the lazy dog"


@pytest.fixture(scope="session")
def inspector():
    cfg = InspectorConfig(model_name="gpt2", device="auto")
    insp = ModelInspector(cfg)
    insp.load()
    return insp


@pytest.fixture(scope="session")
def run_result(inspector):
    logits, cache = inspector.run(PROMPT)
    return logits, cache


# ------------------------------------------------------------------
# Load
# ------------------------------------------------------------------

def test_load_gpt2(inspector):
    assert inspector.model is not None


def test_model_properties(inspector):
    assert inspector.n_layers == 12
    assert inspector.n_heads == 12
    assert inspector.d_model == 768


# ------------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------------

def test_run_returns_logits_shape(run_result, inspector):
    logits, _ = run_result
    seq_len = len(inspector.to_str_tokens(PROMPT))
    assert logits.ndim == 3
    assert logits.shape[0] == 1
    assert logits.shape[1] == seq_len
    assert logits.shape[2] == 50257  # GPT-2 vocab size


def test_run_cache_not_empty(run_result):
    _, cache = run_result
    assert len(list(cache.keys())) > 0


# ------------------------------------------------------------------
# Attention patterns
# ------------------------------------------------------------------

def test_attention_patterns_shape(run_result, inspector):
    logits, cache = run_result
    seq_len = logits.shape[1]
    patterns = inspector.get_attention_patterns(cache, layer=0)
    assert patterns.shape == (inspector.n_heads, seq_len, seq_len)


def test_attention_patterns_sum_to_one(run_result, inspector):
    _, cache = run_result
    patterns = inspector.get_attention_patterns(cache, layer=0)
    row_sums = patterns.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_all_attention_patterns_shape(run_result, inspector):
    logits, cache = run_result
    seq_len = logits.shape[1]
    all_patterns = inspector.get_all_attention_patterns(cache)
    assert all_patterns.shape == (
        inspector.n_layers, inspector.n_heads, seq_len, seq_len
    )


# ------------------------------------------------------------------
# Residual stream
# ------------------------------------------------------------------

def test_residual_stream_shape(run_result, inspector):
    logits, cache = run_result
    seq_len = logits.shape[1]
    residual = inspector.get_residual_stream(cache)
    assert residual.shape == (inspector.n_layers + 1, seq_len, inspector.d_model)


def test_residual_stream_layer_count(run_result, inspector):
    _, cache = run_result
    residual = inspector.get_residual_stream(cache)
    # n_layers + 1 (embedding) = 13 for GPT-2 small
    assert residual.shape[0] == inspector.n_layers + 1


# ------------------------------------------------------------------
# Logit lens
# ------------------------------------------------------------------

def test_logit_lens_length(run_result, inspector):
    _, cache = run_result
    lens = inspector.get_logit_lens(cache, top_k=5)
    assert len(lens) == inspector.n_layers + 1


def test_logit_lens_top_k(run_result, inspector):
    _, cache = run_result
    for k in (1, 5, 10):
        lens = inspector.get_logit_lens(cache, top_k=k)
        for layer_result in lens:
            assert len(layer_result) == k


def test_logit_lens_probs_sum_approx_one(run_result, inspector):
    _, cache = run_result
    lens = inspector.get_logit_lens(cache, top_k=50257)
    total = sum(p for _, p in lens[0])
    assert abs(total - 1.0) < 0.01


# ------------------------------------------------------------------
# Neuron activations
# ------------------------------------------------------------------

def test_neuron_activations_shape(run_result, inspector):
    logits, cache = run_result
    seq_len = logits.shape[1]
    acts = inspector.get_neuron_activations(cache, layer=0)
    d_mlp = inspector.model.cfg.d_mlp
    assert acts.shape == (seq_len, d_mlp)


# ------------------------------------------------------------------
# Error handling
# ------------------------------------------------------------------

def test_requires_load():
    insp = ModelInspector("gpt2")
    with pytest.raises(RuntimeError, match="load"):
        insp.run("hello")
