from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import torch
from torch import Tensor

log = logging.getLogger("neuroscope")
if not log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    log.addHandler(_h)
    log.setLevel(logging.DEBUG)


@dataclass
class InspectorConfig:
    model_name: str
    device: str = "auto"
    center_writing_weights: bool = True
    fold_ln: bool = True


class ModelInspector:
    """
    Thin wrapper over HookedTransformer that standardises access to activations,
    attention patterns, residual stream states and logit-lens projections.
    HookedTransformer is imported lazily inside load() to avoid Streamlit startup crashes.
    """

    def __init__(self, config: InspectorConfig | str):
        if isinstance(config, str):
            config = InspectorConfig(model_name=config)
        self.config = config
        self.model = None  # type: ignore[assignment]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        import faulthandler
        import threading
        faulthandler.enable()  # dumps C-level traceback to stderr on segfault/SIGABRT
        log.info("=== load() start: model=%s ===", self.config.model_name)
        log.info("thread=%s  is_main=%s", threading.current_thread().name, threading.current_thread() is threading.main_thread())

        log.debug("importing transformer_lens...")
        try:
            from transformer_lens import HookedTransformer  # noqa: PLC0415
        except BaseException as exc:
            log.error("CRASH during import: %s: %s", type(exc).__name__, exc, exc_info=True)
            raise
        log.debug("transformer_lens imported OK")

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Device selected: %s (CUDA available: %s)", device, torch.cuda.is_available())

        log.info("Calling HookedTransformer.from_pretrained('%s')...", self.config.model_name)
        self.model = HookedTransformer.from_pretrained(
            self.config.model_name,
            center_writing_weights=self.config.center_writing_weights,
            fold_ln=self.config.fold_ln,
            device=device,
        )
        log.info("Model loaded. Switching to eval mode...")
        self.model.eval()
        log.info("=== load() done ===")


    def _require_model(self):
        if self.model is None:
            raise RuntimeError("Call .load() before using the inspector.")
        return self.model

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def run(self, prompt: str) -> tuple[Tensor, dict]:
        """
        Returns (logits, cache).
        logits shape: [1, seq_len, vocab_size]
        """
        model = self._require_model()
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        return logits, cache

    def run_tokens(self, tokens: Tensor) -> tuple[Tensor, dict]:
        model = self._require_model()
        with torch.no_grad():
            logits, cache = model.run_with_cache(tokens)
        return logits, cache

    # ------------------------------------------------------------------
    # Attention patterns
    # ------------------------------------------------------------------

    def get_attention_patterns(self, cache: dict, layer: int) -> Tensor:
        """Shape: [n_heads, seq_len, seq_len]"""
        return cache["pattern", layer].squeeze(0)

    def get_all_attention_patterns(self, cache: dict) -> Tensor:
        """Shape: [n_layers, n_heads, seq_len, seq_len]"""
        model = self._require_model()
        return torch.stack(
            [cache["pattern", layer].squeeze(0) for layer in range(model.cfg.n_layers)]
        )

    # ------------------------------------------------------------------
    # Residual stream
    # ------------------------------------------------------------------

    def get_residual_stream(self, cache: dict) -> Tensor:
        """
        Shape: [n_layers + 1, seq_len, d_model]
        Index 0 = embedding, Index L = after layer L-1.
        """
        model = self._require_model()
        embed = (cache["hook_embed"] + cache["hook_pos_embed"]).squeeze(0)
        states = [embed]
        for layer in range(model.cfg.n_layers):
            states.append(cache["resid_post", layer].squeeze(0))
        return torch.stack(states)

    # ------------------------------------------------------------------
    # Logit lens
    # ------------------------------------------------------------------

    def get_logit_lens(self, cache: dict, top_k: int = 10) -> list[list[tuple[str, float]]]:
        """
        Top-K token predictions at each layer for the last sequence position.
        Returns list[layer] of list[(token_str, probability)].
        """
        model = self._require_model()
        residual = self.get_residual_stream(cache)  # [n_layers+1, seq, d_model]
        last_pos = residual[:, -1, :]               # [n_layers+1, d_model]

        result: list[list[tuple[str, float]]] = []
        for state in last_pos:
            x = state.unsqueeze(0).unsqueeze(0)      # [1, 1, d_model]
            normed = model.ln_final(x)               # [1, 1, d_model]
            logits = model.unembed(normed).squeeze() # [vocab_size]
            probs = torch.softmax(logits, dim=-1)
            top_probs, top_ids = probs.topk(top_k)
            tokens = [
                (model.tokenizer.decode([int(idx)]), float(p))
                for idx, p in zip(top_ids.tolist(), top_probs.tolist())
            ]
            result.append(tokens)

        return result

    # ------------------------------------------------------------------
    # Neuron activations
    # ------------------------------------------------------------------

    def get_neuron_activations(self, cache: dict, layer: int) -> Tensor:
        """Shape: [seq_len, d_mlp]"""
        return cache["post", layer].squeeze(0)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def n_layers(self) -> int:
        return self._require_model().cfg.n_layers

    @property
    def n_heads(self) -> int:
        return self._require_model().cfg.n_heads

    @property
    def d_model(self) -> int:
        return self._require_model().cfg.d_model

    def to_tokens(self, text: str) -> Tensor:
        return self._require_model().to_tokens(text)

    def to_str_tokens(self, text: str) -> list[str]:
        return self._require_model().to_str_tokens(text)
