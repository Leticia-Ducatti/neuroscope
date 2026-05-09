"""
Step 0.3 smoke test — verifica que o stack está funcional antes de qualquer código de produto.
Executar com: python scripts/smoke_test.py
"""

import sys


def check(label: str, fn):
    try:
        result = fn()
        print(f"  [OK] {label}: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] {label}: {e}")
        return False


def main():
    print("=== NeuroScope Smoke Test ===\n")
    failures = 0

    # 1. Python version
    ok = check(
        "Python >= 3.11",
        lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    if not ok:
        failures += 1

    # 2. Torch
    def torch_check():
        import torch
        cuda = torch.cuda.is_available()
        return f"version={torch.__version__}, CUDA={cuda}"

    if not check("PyTorch", torch_check):
        failures += 1

    # 3. TransformerLens
    def tl_check():
        import importlib.metadata
        import transformer_lens  # noqa: F401
        version = importlib.metadata.version("transformer_lens")
        return f"version={version}"

    if not check("TransformerLens", tl_check):
        failures += 1

    # 4. Transformers
    def hf_check():
        import transformers
        return f"version={transformers.__version__}"

    if not check("HuggingFace Transformers", hf_check):
        failures += 1

    # 5. Streamlit
    def st_check():
        import streamlit
        return f"version={streamlit.__version__}"

    if not check("Streamlit", st_check):
        failures += 1

    # 6. Plotly
    def plotly_check():
        import plotly
        return f"version={plotly.__version__}"

    if not check("Plotly", plotly_check):
        failures += 1

    # 7. einops
    def einops_check():
        import einops
        return f"version={einops.__version__}"

    if not check("einops", einops_check):
        failures += 1

    # 8. End-to-end: carrega GPT-2 e faz forward pass
    print("\n  [Loading GPT-2 small for end-to-end test — pode demorar no primeiro run...]")

    def e2e_check():
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained("gpt2")
        tokens = model.to_tokens("Hello, world!")
        logits, cache = model.run_with_cache(tokens)

        assert logits.ndim == 3, f"expected 3D logits, got {logits.ndim}D"
        assert logits.shape[-1] == 50257, f"unexpected vocab size {logits.shape[-1]}"
        assert len(list(cache.keys())) > 0, "cache is empty"

        return f"logits={tuple(logits.shape)}, cache_keys={len(list(cache.keys()))}"

    if not check("GPT-2 end-to-end", e2e_check):
        failures += 1

    print(f"\n{'=' * 32}")
    if failures == 0:
        print("Todos os checks passaram. Ambiente pronto para a Fase 1.")
    else:
        print(f"{failures} check(s) falharam. Resolva antes de continuar.")
        sys.exit(1)


if __name__ == "__main__":
    main()
