# NeuroScope 🔬

**Interactive mechanistic interpretability for transformer language models.**

Inspect the internal computations of GPT-2 and Pythia — attention patterns, residual stream evolution, logit lens projections, and causal circuit analysis — all through a visual Streamlit interface.

---

## Pages

| Page | What it shows |
|------|---------------|
| **Attention Explorer** | Softmax attention weights per layer and head |
| **Residual Stream** | L2 norm growth and PCA trajectory across layers |
| **Logit Lens** | Layer-by-layer token predictions before the final output |
| **Induction Heads** | Prefix matching and copying scores (Olsson et al., 2022) |
| **IOI Circuit** | Activation patching map for indirect object identification (Wang et al., 2022) |

---

## Quickstart

**Requires Python 3.11** (Python 3.13 has a Windows bug that crashes `transformer_lens` inside Streamlit threads).

```bash
# 1. Clone and enter the directory
git clone <repo-url>
cd neuroscope

# 2. Create a virtual environment with Python 3.11
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
# source .venv/bin/activate         # Linux / macOS

# 3. Install PyTorch — pick ONE:

# Option A: No GPU (CPU only, slower but works on any machine)
pip install torch torchvision

# Option B: NVIDIA GPU with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# 4. Install remaining dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run the app
.venv\Scripts\python.exe -m streamlit run app.py
```

Open **http://localhost:8501**, load a model from the sidebar, and start exploring.

---

## Supported Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| `gpt2` | 124M | Default — fastest to load |
| `gpt2-medium` | 355M | — |
| `EleutherAI/pythia-160m` | 160M | — |
| `EleutherAI/pythia-1.4b` | 1.4B | Requires ≥ 8GB RAM |

Model weights are cached in `~/.cache/huggingface/` after the first download.

---

## Requirements

- Python 3.11+
- 8 GB RAM minimum (16 GB recommended for larger models)
- NVIDIA GPU with CUDA 12.1+ (optional — CPU works for GPT-2 small)

---

## Project Structure

```
app.py                  # Entry point
pages/                  # Streamlit pages (one per analysis)
src/
  inspector/            # ModelInspector — unified access to activations and weights
  circuits/             # Circuit utilities: induction heads, IOI patching
  viz/                  # Shared UI helpers (session state, CSS, i18n)
data/prompts/           # Standardised test prompts (IOI dataset)
notebooks/              # Exploratory analysis
tests/                  # Unit tests
```

---

## Roadmap

| Phase | Model | Goal |
|-------|-------|------|
| ✅ 1 | GPT-2 small | Induction heads, IOI circuit, Streamlit app |
| 🔲 2 | Pythia 1.4B | Circuit persistence at larger scale |
| 🔲 3 | Llama 3 8B (4-bit) | Activation steering with A/B interface |

---

## References

- Elhage et al. (2021) — [A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html)
- Olsson et al. (2022) — [In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)
- Wang et al. (2022) — [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593)
- Rimsky et al. (2023) — [Steering Llama 2 via Contrastive Activation Addition](https://arxiv.org/abs/2312.06681)

# NeuroScope

Interactive mechanistic interpretability framework for transformer language models.

NeuroScope enables inspection of internal computations in GPT-2 and Pythia models, including attention patterns, residual stream dynamics, logit lens projections, and causal circuit analysis through an interactive Streamlit interface.

---

## Features

| Module | Description |
|--------|-------------|
| Attention Explorer | Visualizes attention distributions across layers and heads |
| Residual Stream | Tracks activation growth and PCA trajectories across layers |
| Logit Lens | Projects intermediate layer representations into token space |
| Induction Heads | Detects pattern matching and copying behavior (Olsson et al., 2022) |
| IOI Circuit | Performs activation patching for indirect object identification (Wang et al., 2022) |

---

## Quickstart

Requires Python 3.11 (some dependencies may be unstable on Python 3.13).

```bash
# Clone repository
git clone <repo-url>
cd neuroscope

# Create virtual environment
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
source .venv/bin/activate     # Linux/macOS

# Install PyTorch (choose one)
# CPU only
pip install torch torchvision

# OR GPU (CUDA 12.6)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run application
streamlit run app.py
```

Open http://localhost:8501 in your browser.

---

## Supported Models

| Model | Parameters | Notes |
|------|-----------|------|
| GPT-2 | 124M | Default lightweight model |
| GPT-2 Medium | 355M | Balanced performance |
| Pythia 160M | 160M | EleutherAI model |
| Pythia 1.4B | 1.4B | Requires higher memory |

Models are automatically downloaded and cached via Hugging Face.

---

## Project Structure

```
app.py
pages/
src/
  inspector/
  circuits/
  viz/
data/prompts/
notebooks/
tests/
```

---

## Roadmap

| Phase | Focus |
|------|------|
| GPT-2 | Induction heads and IOI circuits |
| Pythia 1.4B | Scaling circuit analysis |
| LLaMA 3 | Activation steering and control interfaces |

---

## References

- Elhage et al. (2021) — Transformer Circuits framework
- Olsson et al. (2022) — Induction heads and in-context learning
- Wang et al. (2022) — IOI circuit analysis
- Rimsky et al. (2023) — Activation steering in LLMs