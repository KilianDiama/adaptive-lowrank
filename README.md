# adaptive-lowrank

Drop-in `nn.Linear` / `nn.Conv2d` layers that **auto-tune active rank** using **Frequent Directions (FD)** on the **inputs**.  
No module re-instantiation (optimizer / hooks / `torch.compile` friendly). **Freeze** for eval, then **materialize** to a fixed dense model for simple, stable serving.

<p align="center">
  <img alt="rank adaptive" src="https://img.shields.io/badge/rank-adaptive-blue">
  <img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.x-red">
  <img alt="license" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Why this exists

- **Lower cost / greener inference**: adapt rank to the actual input subspace during training or long-running inference ⇒ **FLOPs ↓, VRAM ↓, latency ↓** with minimal accuracy impact.
- **Operations-friendly**: rank changes do **not** recreate submodules; optimizers, hooks, and `torch.compile` remain stable.
- **Production posture**: after adaptation, **materialize** to plain dense `Linear`/`Conv2d`—deploy anywhere that runs PyTorch/TorchScript/ONNX.

---

## Key features

- **AdaptiveLinear / AdaptiveConv2d** with **max-rank allocation** and **active-rank slicing** (no re-instantiation).
- **Frequent Directions** sketch on **inputs** (robust in FP32) + **hysteresis & cooldown** for stable rank changes.
- **Warm init**: SVD init for the head; Kaiming for the inactive tail.
- **Input/patch sampling** + per-update **row caps** for FD.
- **Persistence**: buffers for rank, steps, and optional FD sketch; config saved via `extra_state`.
- **QoL controls**:
  - `with m.adapt_disabled(): ...` — temporarily freeze adaptation
  - `m.adapt_now()` — force an immediate adaptation check
  - `materialize_model()` — export a fixed dense model

---

## Install

```bash
# Clone this repo
git clone https://github.com/<you>/adaptive-lowrank.git
cd adaptive-lowrank

# (Optional) venv
python -m venv .venv && source .venv/bin/activate

# Dependencies for the demo & integrations
pip install -U torch torchvision timm fastapi uvicorn onnx
You can also vendor the single-file module into your project if you prefer.

Quick Start (Green Serving Demo)
The repo includes a monolithic demo: examples/green_serving_demo.py
It wraps a model, lets ranks adapt briefly, measures latency/throughput/VRAM/accuracy before/after/materialized, and exports TorchScript/ONNX.

bash

# ResNet18, quick benchmark with real CIFAR-10
python examples/green_serving_demo.py --model resnet18 --epochs 1 --eps 0.08 --r-max 128

# No data (synthetic) — fastest sanity check
python examples/green_serving_demo.py --model resnet18 --epochs 0 --no-data --batch 256

# timm models (e.g., ViT tiny) + torch.compile
python examples/green_serving_demo.py --model timm_vit_tiny_patch16_224 --epochs 1 --compile
Artifacts saved to artifacts/:

adaptive_state_dict.pt, dense_state_dict.pt

model_scripted.pt (TorchScript), model.onnx (if ONNX available)

bench_summary.csv (latency/throughput/VRAM summary)

Use in your code (2 lines)
python

import timm
from adaptive_lowrank.core import wrap_model_with_adaptive, materialize_model, iter_adaptive_layers

model = timm.create_model("resnet50", pretrained=False, num_classes=1000)
model = wrap_model_with_adaptive(model, eps=0.08, r_max=128, include_conv=True)

# ...train a bit to let ranks adapt...

dense = materialize_model(model).eval()  # fixed dense model for serving
API
wrap_model_with_adaptive(model, **kwargs) -> nn.Module
Key kwargs:

eps=0.08 – target reconstruction error for FD; controls rank.

r_min=4, r_max=128 – rank bounds.

adapt_every=50 – adaptation cadence (in forward steps).

grow_step=16, shrink_step=32 – rank change per step.

sketch_size=256 – FD sketch rows (≥ r_max recommended).

include_conv=True – also wrap Conv2d (groups=1).

freeze_during_eval=True, adapt_cooldown=10, sample_stride=1, max_rows_fd=2048.

svd_backend="auto" – choose SVD device.

Layer helpers
m.current_rank()

m.enable_adaptation(True|False)

with m.adapt_disabled(): ...

m.adapt_now()

Model helpers
materialize_model(model) -> nn.Module – deep-copies model with dense Linear/Conv2d.

iter_adaptive_layers(model) – iterate all adaptive layers.

set_rank_change_callback(model, cb) – callback (old_rank, new_rank).

clear_fd_sketch(model) – reset FD matrices & counters.

How it works (brief)
We allocate max-rank weights once:

Linear: B: in→r_max (bias=False) then A: r_max→out (bias=True); forward uses first r_active columns.

Conv2d: conv1: in→r_max, k×k, bias=False then conv2: r_max→out, 1×1, bias=True.
FD sketches the input rows/patches in FP32 and estimates how many singular directions capture (1−ε) of energy → choose r_active.
No module is recreated when rank changes—only slicing—so optimizers and torch.compile are stable.
When ready for prod, materialize back to a conventional dense model.

Integrations
timm (vision): wrap any ResNet/ViT; log active ranks each epoch.

Hugging Face (Linear-only first): wrap nn.Linear modules; fine-tune briefly, then materialize.

PyTorch Lightning: small callback to log ranks/FD energy.

Serving: FastAPI example in the demo; use materialized model for a non-mutable, ops-friendly artifact.

torch.compile: supported on the wrapped/model (and on the materialized model).

Example results (template — replace with your runs)
Setup	Acc@1	Latency ↓	Throughput ↑	VRAM Peak ↓	Mean Rank
Dense ResNet18	95.1%	1.00×	1.00×	1.00×	–
Adaptive (ε=0.08)	95.0%	0.75×	1.35×	0.70×	~56
Materialized (fixed)	95.0%	0.72×	1.38×	0.68×	fixed

Numbers above are illustrative; run the demo to produce your own.

Tips & knobs
Target quality: start with eps=0.08, r_max equal to a comfortable cap, and adapt_cooldown=10.

Stability: if ranks oscillate, increase adapt_cooldown or adapt_every.

Speed: use --compile (PyTorch 2.x). Materialized models often compile even better.

Conv caveat: groups != 1 not supported (yet).

AMP: FD runs in FP32 by design for stability; model weights keep their dtype.

Security & safety
⚠️ torch.load uses Python pickle. Never load untrusted files.

FD operates on inputs; ensure your privacy policy allows runtime statistics. You can disable adaptation in eval() and/or via adapt_disabled().

Roadmap
Conv2d with groups>1 (depthwise/separable).

Optional randomized SVD path for very large d.

First-class logging hooks (W&B / MLflow).

Ready-made Lightning & HF callbacks in callbacks/.

Contributing
Issues and PRs are welcome! Please include:

a minimal repro or failing unit test,

PyTorch / CUDA versions,

device details (CPU/GPU).

Run tests locally before PR:

bash

pytest -q
Citation
If you use this project in research, please cite the original Frequent Directions work and this repository.

Frequent Directions: D. Liberty, Simple and Deterministic Matrix Sketching, 2013.

This repo: <Your Name>, adaptive-lowrank, 2025. GitHub: https://github.com/<you>/adaptive-lowrank

arduino

@misc{adaptive_lowrank_2025,
  title  = {adaptive-lowrank: Adaptive rank layers for PyTorch using Frequent Directions},
  author = {<Your Name>},
  year   = {2025},
  url    = {https://github.com/<you>/adaptive-lowrank}
}
License
MIT — see LICENSE.
