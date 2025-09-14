#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Green Serving Demo – AdaptiveLowRank (monolithic, polished)
-----------------------------------------------------------
- Wrap de modèles vision (torchvision/timm) avec couches à rang adaptatif.
- Bench rapide: accuracy, latence, throughput, VRAM (CUDA si dispo) — mesures corrigées.
- Matérialisation dense + export TorchScript (+ ONNX si installé).
- Option de service FastAPI avec le modèle matérialisé.
- Fonctionne offline avec --no-data (données synthétiques).

Exemples:
  python green_serving_demo.py --model resnet18 --epochs 1 --eps 0.08 --r-max 128
  python green_serving_demo.py --model timm_vit_tiny_patch16_224 --epochs 1 --compile --no-data --save-out outdir

Notes:
- Si pas d'Internet/dataset: utilisez --no-data pour benchmark synthétique.
- FastAPI/uvicorn ne sont requis que si --serve.
"""

from __future__ import annotations

import argparse
import math
import os
import time
import warnings
from dataclasses import asdict, dataclass
from typing import Optional, Tuple, Dict, Any, Callable, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

# ===============================
# AdaptiveLowRank (version augmentée)
# ===============================

class FrequentDirections:
    """
    FD sketch d'un flux X ∈ R^{n×d}, conserve B ∈ R^{ℓ×d} (~top directions).
    Améliorations:
      - SVD robuste GPU->CPU.
      - Filtrage des lignes non finies.
      - Workspace Z réutilisable pour réduire les allocs.
    """
    def __init__(
        self,
        d: int,
        ell: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        svd_backend: str = "auto",  # "auto" | "gpu" | "cpu"
    ):
        assert ell > 0 and d > 0
        self.d = int(d)
        self.ell = int(ell)
        self.device = device
        self.dtype = dtype
        self.svd_backend = svd_backend
        self.B = torch.zeros((self.ell, self.d), device=device, dtype=dtype)
        self.n_seen = 0
        self._Z_workspace: Optional[torch.Tensor] = None  # pré-allocation souple

    def _svd(self, Z: torch.Tensor):
        backend = self.svd_backend
        if backend == "gpu":
            return torch.linalg.svd(Z, full_matrices=False)
        if backend == "cpu":
            U, S, Vh = torch.linalg.svd(Z.cpu(), full_matrices=False)
            return U.to(self.device), S.to(self.device), Vh.to(self.device)
        try:
            return torch.linalg.svd(Z, full_matrices=False)
        except RuntimeError as e:
            warnings.warn(f"[FD] svd GPU failed ({e}); fallback CPU.", RuntimeWarning)
            U, S, Vh = torch.linalg.svd(Z.detach().cpu(), full_matrices=False)
            return U.to(self.device), S.to(self.device), Vh.to(self.device)

    @torch.no_grad()
    def update(self, X: torch.Tensor):
        if X is None or X.numel() == 0:
            return
        X = X.to(self.device, dtype=self.dtype)
        if X.shape[0] == 0:
            return
        if not torch.isfinite(X).all():
            mask = torch.isfinite(X).all(dim=1)
            X = X[mask]
            if X.shape[0] == 0:
                return

        need_rows = self.ell + X.shape[0]
        if (self._Z_workspace is None
            or self._Z_workspace.shape[0] < need_rows
            or self._Z_workspace.shape[1] != self.d
            or self._Z_workspace.dtype != self.dtype
            or self._Z_workspace.device != self.device):
            self._Z_workspace = torch.empty((need_rows, self.d), device=self.device, dtype=self.dtype)

        Z = self._Z_workspace[:need_rows, :]
        Z[:self.ell].copy_(self.B)
        Z[self.ell:need_rows].copy_(X)

        _, S, Vh = self._svd(Z)
        if S.numel() < self.ell:
            self.B.zero_()
            Bnew = (S.unsqueeze(1) * Vh)
            self.B[:Bnew.shape[0], :] = Bnew
        else:
            tau = (S[self.ell - 1] ** 2).clamp(min=0.0)
            S2 = (S ** 2 - tau).clamp(min=0.0)
            S_shrunk = torch.sqrt(S2 + 1e-12)
            S_top = S_shrunk[: self.ell]
            Vh_top = Vh[: self.ell, :]
            self.B = (S_top.unsqueeze(1) * Vh_top).contiguous()

        self.n_seen += int(X.shape[0])

    @torch.no_grad()
    def energy_profile(self) -> Tuple[torch.Tensor, float]:
        if self.B.numel() == 0 or torch.count_nonzero(self.B) == 0:
            empty = torch.zeros(0, device=self.device, dtype=self.dtype)
            return empty, 0.0
        try:
            Sb = torch.linalg.svdvals(self.B)
        except RuntimeError:
            Sb = torch.linalg.svdvals(self.B.cpu()).to(self.device)
        S2 = Sb ** 2
        return S2, float(S2.sum().item())

    @torch.no_grad()
    def choose_rank_for_eps(self, eps: float, r_min: int, r_max: int) -> int:
        S2, tot = self.energy_profile()
        if S2.numel() == 0 or tot <= 0:
            return max(r_min, 1)
        cumsum = torch.cumsum(S2, dim=0)
        resid = torch.clamp(tot - cumsum, min=0.0) * 0.9995
        thresh = (eps ** 2) * tot
        if (resid <= thresh).any():
            r_star = int(resid.le(thresh).nonzero(as_tuple=False).min().item() + 1)
        else:
            r_star = S2.numel()
        return int(max(r_min, min(r_star, r_max)))

    def state_dict(self) -> dict:
        return {
            "d": int(self.d),
            "ell": int(self.ell),
            "B": self.B,
            "n_seen": int(self.n_seen),
            "dtype": str(self.dtype),
            "device": str(self.device),
            "svd_backend": str(self.svd_backend),
        }

    def load_state_dict(self, state: dict):
        B = state.get("B", None)
        if B is not None:
            self.B = B.to(self.device, dtype=self.dtype).clone()
        self.n_seen = int(state.get("n_seen", 0))
        self.svd_backend = state.get("svd_backend", "auto")


@dataclass
class AdaptConfig:
    eps_grow: float = 0.07
    eps_shrink: float = 0.09
    r_min: int = 4
    r_max: int = 128
    sketch_size: int = 256
    adapt_every: int = 50
    grow_step: int = 16
    shrink_step: int = 32
    track_inputs_every: int = 1
    freeze_during_eval: bool = True
    sample_stride: int = 1
    max_rows_fd: int = 2048
    warmup_steps: int = 0
    svd_backend: str = "auto"
    persist_sketch: bool = True
    min_rows_fd: int = 0
    adapt_cooldown: int = 0
    disable_amp_for_fd: bool = True


class _AdaptiveBase(nn.Module):
    def __init__(self):
        super().__init__()
        self._fd_initialized = False
        self.on_rank_change: Optional[Callable[[int, int], None]] = None
        self.adapt_enabled: bool = True

    def get_extra_state(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if hasattr(self, "cfg") and isinstance(self.cfg, AdaptConfig):
            d["adapt_config"] = asdict(self.cfg)
        d["adapt_enabled"] = bool(getattr(self, "adapt_enabled", True))
        return d

    def set_extra_state(self, state: Dict[str, Any]):
        cfg_dict = state.get("adapt_config")
        if cfg_dict is not None and hasattr(self, "cfg"):
            for k, v in cfg_dict.items():
                if hasattr(self.cfg, k):
                    setattr(self.cfg, k, v)
        if "adapt_enabled" in state:
            self.adapt_enabled = bool(state["adapt_enabled"])

    def _register_fd_buffers(self, fd: FrequentDirections):
        persist = True
        if hasattr(self, "cfg") and isinstance(self.cfg, AdaptConfig):
            persist = bool(self.cfg.persist_sketch)
        self.register_buffer("fd_B", fd.B.clone(), persistent=persist)
        self.register_buffer("fd_n_seen", torch.tensor(fd.n_seen, dtype=torch.int64), persistent=persist)
        self._fd_initialized = True
        self.register_buffer("_last_adapt_step", torch.tensor(-10**9, dtype=torch.int64), persistent=True)

    @torch.no_grad()
    def _sync_fd_to_buffers(self):
        if not self._fd_initialized:
            return
        if self.fd_B.shape != self.fd.B.shape or self.fd_B.dtype != self.fd.B.dtype or self.fd_B.device != self.fd.B.device:
            persist = True
            if hasattr(self, "cfg") and isinstance(self.cfg, AdaptConfig):
                persist = bool(self.cfg.persist_sketch)
            self.register_buffer("fd_B", torch.zeros_like(self.fd.B), persistent=persist)
        self.fd_B.copy_(self.fd.B)
        self.fd_n_seen.fill_(int(self.fd.n_seen))

    @torch.no_grad()
    def _sync_buffers_to_fd(self):
        if not self._fd_initialized:
            return
        if self.fd.B.shape != self.fd_B.shape or self.fd.B.device != self.fd_B.device or self.fd.B.dtype != self.fd_B.dtype:
            self.fd.B = torch.zeros_like(self.fd_B)
        self.fd.B.copy_(self.fd_B)
        self.fd.device = self.fd_B.device
        self.fd.n_seen = int(self.fd_n_seen.item())

    def to(self, *args, **kwargs):
        mod = super().to(*args, **kwargs)
        if hasattr(self, "_sync_buffers_to_fd"):
            self._sync_buffers_to_fd()
        return mod

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        if hasattr(self, "fd"):
            self._sync_buffers_to_fd()

    @torch.no_grad()
    def get_rank_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            "rank_active": int(getattr(self, "rank_active", torch.tensor(-1)).item()),
            "rank_max": int(getattr(self, "rank_max", -1)),
            "steps": int(getattr(self, "_step", torch.tensor(0)).item()),
        }
        fd = getattr(self, "fd", None)
        if fd is not None:
            stats["n_seen"] = int(fd.n_seen)
            S2, tot = fd.energy_profile()
            stats["energy_total"] = float(tot)
            stats["energy_prefix"] = [float(v) for v in S2.tolist()]
        else:
            stats["n_seen"] = 0
        return stats

    def enable_adaptation(self, enabled: bool = True):
        self.adapt_enabled = bool(enabled)

    # Contexte pratique pour geler temporairement l’adaptation
    from contextlib import contextmanager
    @contextmanager
    def adapt_disabled(self):
        prev = self.adapt_enabled
        try:
            self.adapt_enabled = False
            yield
        finally:
            self.adapt_enabled = prev

    # Garder FD en fp32 si AMP actif
    def _fd_autocast_guard(self):
        if getattr(self, "cfg", None) is None or not self.cfg.disable_amp_for_fd:
            class _Noop:
                def __enter__(self_s): return None
                def __exit__(self_s, exc_type, exc, tb): return False
            return _Noop()
        device_type = getattr(self, "fd_B", None).device.type if hasattr(self, "fd_B") else ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            return torch.amp.autocast(device_type=device_type, enabled=False)
        except Exception:
            if device_type == "cuda" and hasattr(torch, "cuda") and hasattr(torch.cuda, "amp"):
                return torch.cuda.amp.autocast(enabled=False)
            class _Noop:
                def __enter__(self_s): return None
                def __exit__(self_s, exc_type, exc, tb): return False
            return _Noop()


class AdaptiveLinear(_AdaptiveBase):
    def __init__(self, base: nn.Linear, cfg: AdaptConfig):
        super().__init__()
        assert isinstance(base, nn.Linear)
        in_f, out_f = base.in_features, base.out_features
        dev = base.weight.device
        dtype = base.weight.dtype

        self.in_features = in_f
        self.out_features = out_f
        self.cfg = cfg

        r_max = int(min(cfg.r_max, min(in_f, out_f)))
        ell = max(cfg.sketch_size, r_max)
        if cfg.sketch_size < r_max:
            warnings.warn(f"[AdaptiveLinear] sketch_size ({cfg.sketch_size}) < r_max ({r_max}); using ell={ell}.", RuntimeWarning)

        r0 = int(max(cfg.r_min, min(min(in_f, out_f, cfg.r_min * 2), r_max)))
        r0 = max(1, min(r0, r_max))
        self.rank_max = r_max
        self.register_buffer("rank_active", torch.tensor(r0, dtype=torch.int32))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.int64))

        self.B_full = nn.Linear(in_f, r_max, bias=False, device=dev, dtype=dtype)
        self.A_full = nn.Linear(r_max, out_f, bias=True, device=dev, dtype=dtype)

        with torch.no_grad():
            W = base.weight.data.to(torch.float32)
            try:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(W.cpu(), full_matrices=False)
                U, S, Vh = U.to(dev), S.to(dev), Vh.to(dev)
            r_init = min(int(self.rank_active.item()), S.numel(), r_max)
            Ur = U[:, :r_init]
            Sr = S[:r_init]
            Vr = Vh[:r_init, :].T
            self.B_full.weight[:r_init].copy_(Vr.T.to(dtype))
            self.A_full.weight[:, :r_init].copy_((Ur * Sr.unsqueeze(0)).to(dtype))
            if base.bias is not None:
                self.A_full.bias.copy_(base.bias.data.to(dtype))
            else:
                nn.init.zeros_(self.A_full.bias)
            if r_init < r_max:
                nn.init.kaiming_uniform_(self.B_full.weight[r_init:], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.A_full.weight[:, r_init:], a=math.sqrt(5))

        self.fd = FrequentDirections(d=in_f, ell=ell, device=dev, dtype=torch.float32, svd_backend=cfg.svd_backend)
        self._register_fd_buffers(self.fd)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank_active={int(self.rank_active.item())}, rank_max={self.rank_max}, "
                f"eps_grow={self.cfg.eps_grow}, eps_shrink={self.cfg.eps_shrink}, "
                f"adapt_every={self.cfg.adapt_every}, warmup_steps={self.cfg.warmup_steps}")

    def _should_track(self) -> bool:
        if not self.adapt_enabled:
            return False
        if (not self.training) and self.cfg.freeze_during_eval:
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_track() and (int(self._step.item()) % self.cfg.track_inputs_every) == 0:
            with torch.no_grad(), self._fd_autocast_guard():
                xb = x.detach()
                X = xb.reshape(-1, xb.shape[-1]).to(torch.float32)
                if not torch.isfinite(X).all():
                    mask = torch.isfinite(X).all(dim=1)
                    X = X[mask]
                if self.cfg.min_rows_fd <= 0 or X.shape[0] >= self.cfg.min_rows_fd:
                    if X.shape[0] > self.cfg.max_rows_fd:
                        idx = torch.randperm(X.shape[0], device=X.device)[: self.cfg.max_rows_fd]
                        X = X.index_select(0, idx)
                    if X.shape[0] > 0:
                        self.fd.update(X)
                        self._sync_fd_to_buffers()

        self._step += 1
        if (self._should_track()
            and int(self._step.item()) >= self.cfg.warmup_steps
            and (int(self._step.item()) % self.cfg.adapt_every) == 0
            and (int(self._step.item()) - int(self._last_adapt_step.item()) >= self.cfg.adapt_cooldown)):
            self._maybe_adapt_rank()

        r = int(self.rank_active.item())
        h = F.linear(x, self.B_full.weight[:r, :], bias=None)
        y = F.linear(h, self.A_full.weight[:, :r], bias=self.A_full.bias)
        return y

    @torch.no_grad()
    def _maybe_adapt_rank(self):
        r_grow = int(self.fd.choose_rank_for_eps(self.cfg.eps_grow, self.cfg.r_min, self.rank_max))
        r_shrk = int(self.fd.choose_rank_for_eps(self.cfg.eps_shrink, self.cfg.r_min, self.rank_max))
        r = int(self.rank_active.item())
        if r_grow > r:
            r_target = min(r_grow, self.rank_max)
            r_new = min(r + self.cfg.grow_step, r_target)
            if r_new != r:
                if callable(self.on_rank_change): self.on_rank_change(r, r_new)
                self.rank_active.fill_(int(r_new))
                self._last_adapt_step.copy_(self._step)
            return
        if r_shrk < r:
            r_target = max(r_shrk, max(self.cfg.r_min, 1))
            r_new = max(r - self.cfg.shrink_step, r_target)
            if r_new != r:
                if callable(self.on_rank_change): self.on_rank_change(r, r_new)
                self.rank_active.fill_(int(r_new))
                self._last_adapt_step.copy_(self._step)
            return

    @torch.no_grad()
    def current_rank(self) -> int:
        return int(self.rank_active.item())

    @torch.no_grad()
    def adapt_now(self):
        if self._should_track() and int(self._step.item()) >= self.cfg.warmup_steps:
            self._maybe_adapt_rank()


class AdaptiveConv2d(_AdaptiveBase):
    def __init__(self, base: nn.Conv2d, cfg: AdaptConfig):
        super().__init__()
        assert isinstance(base, nn.Conv2d)
        if base.groups != 1:
            raise AssertionError("groups!=1 not supported by AdaptiveConv2d.")
        kH, kW = base.kernel_size
        cin = base.in_channels
        cout = base.out_channels
        dev = base.weight.device
        dtype = base.weight.dtype

        self.in_channels = cin
        self.out_channels = cout
        self.kernel_size = (int(kH), int(kW))
        self.stride = base.stride
        self.padding = base.padding
        self.dilation = base.dilation
        self.cfg = cfg

        k_elems = cin * kH * kW
        r_max = int(min(cfg.r_max, min(k_elems, cout)))
        ell = max(cfg.sketch_size, r_max)
        if cfg.sketch_size < r_max:
            warnings.warn(f"[AdaptiveConv2d] sketch_size ({cfg.sketch_size}) < r_max ({r_max}); using ell={ell}.", RuntimeWarning)

        r0 = int(max(cfg.r_min, min(min(k_elems, cout, cfg.r_min * 2), r_max)))
        r0 = max(1, min(r0, r_max))
        self.rank_max = r_max
        self.register_buffer("rank_active", torch.tensor(r0, dtype=torch.int32))
        self.register_buffer("_step", torch.tensor(0, dtype=torch.int64))

        self.conv1_full = nn.Conv2d(cin, r_max, kernel_size=self.kernel_size, stride=self.stride,
                                    padding=self.padding, dilation=self.dilation, bias=False, device=dev, dtype=dtype)
        self.conv2_full = nn.Conv2d(r_max, cout, kernel_size=1, bias=True, device=dev, dtype=dtype)

        with torch.no_grad():
            Wm = base.weight.data.to(torch.float32).reshape(cout, k_elems)
            try:
                U, S, Vh = torch.linalg.svd(Wm, full_matrices=False)
            except RuntimeError:
                U, S, Vh = torch.linalg.svd(Wm.cpu(), full_matrices=False)
                U, S, Vh = U.to(dev), S.to(dev), Vh.to(dev)

            r_init = min(int(self.rank_active.item()), S.numel(), r_max)
            Ur = U[:, :r_init]
            Sr = S[:r_init]
            Vr = Vh[:r_init, :].T
            W1 = Vr.T.reshape(r_init, cin, kH, kW).to(dtype)
            self.conv1_full.weight[:r_init].copy_(W1)
            W2 = (Ur * Sr.unsqueeze(0)).reshape(cout, r_init, 1, 1).to(dtype)
            self.conv2_full.weight[:, :r_init].copy_(W2)
            if base.bias is not None:
                self.conv2_full.bias.copy_(base.bias.data.to(dtype))
            else:
                nn.init.zeros_(self.conv2_full.bias)
            if r_init < r_max:
                nn.init.kaiming_uniform_(self.conv1_full.weight[r_init:], a=math.sqrt(5))
                nn.init.kaiming_uniform_(self.conv2_full.weight[:, r_init:], a=math.sqrt(5))

        self.fd = FrequentDirections(d=k_elems, ell=ell, device=dev, dtype=torch.float32, svd_backend=cfg.svd_backend)
        self._register_fd_buffers(self.fd)

    def extra_repr(self) -> str:
        return (f"in_channels={self.in_channels}, out_channels={self.out_channels}, k={self.kernel_size}, "
                f"rank_active={int(self.rank_active.item())}, rank_max={self.rank_max}, "
                f"eps_grow={self.cfg.eps_grow}, eps_shrink={self.cfg.eps_shrink}, "
                f"adapt_every={self.cfg.adapt_every}, warmup_steps={self.cfg.warmup_steps}")

    def _should_track(self) -> bool:
        if not self.adapt_enabled:
            return False
        if (not self.training) and self.cfg.freeze_during_eval:
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._should_track() and (int(self._step.item()) % self.cfg.track_inputs_every) == 0:
            with torch.no_grad(), self._fd_autocast_guard():
                xb = x.detach()
                if self.cfg.sample_stride > 1:
                    xb = xb[:, :, :: self.cfg.sample_stride, :: self.cfg.sample_stride]
                patches = F.unfold(xb, kernel_size=self.kernel_size, dilation=self.dilation,
                                   padding=self.padding, stride=self.stride)
                P = patches.transpose(1, 2).reshape(-1, patches.shape[1]).to(torch.float32)
                if not torch.isfinite(P).all():
                    mask = torch.isfinite(P).all(dim=1)
                    P = P[mask]
                if not (self.cfg.min_rows_fd > 0 and P.shape[0] < self.cfg.min_rows_fd):
                    if P.shape[0] > self.cfg.max_rows_fd:
                        idx = torch.randperm(P.shape[0], device=P.device)[: self.cfg.max_rows_fd]
                        P = P.index_select(0, idx)
                    if P.shape[0] > 0:
                        self.fd.update(P)
                        self._sync_fd_to_buffers()

        self._step += 1
        if (self._should_track()
            and int(self._step.item()) >= self.cfg.warmup_steps
            and (int(self._step.item()) % self.cfg.adapt_every) == 0
            and (int(self._step.item()) - int(self._last_adapt_step.item()) >= self.cfg.adapt_cooldown)):
            self._maybe_adapt_rank()

        r = int(self.rank_active.item())
        y1 = F.conv2d(x, weight=self.conv1_full.weight[:r, :, :, :], bias=None,
                      stride=self.stride, padding=self.padding, dilation=self.dilation)
        y2 = F.conv2d(y1, weight=self.conv2_full.weight[:, :r, :, :], bias=self.conv2_full.bias,
                      stride=1, padding=0, dilation=1)
        return y2

    @torch.no_grad()
    def _maybe_adapt_rank(self):
        r_grow = int(self.fd.choose_rank_for_eps(self.cfg.eps_grow, self.cfg.r_min, self.rank_max))
        r_shrk = int(self.fd.choose_rank_for_eps(self.cfg.eps_shrink, self.cfg.r_min, self.rank_max))
        r = int(self.rank_active.item())
        if r_grow > r:
            r_target = min(r_grow, self.rank_max)
            r_new = min(r + self.cfg.grow_step, r_target)
            if r_new != r:
                if callable(self.on_rank_change): self.on_rank_change(r, r_new)
                self.rank_active.fill_(int(r_new))
                self._last_adapt_step.copy_(self._step)
            return
        if r_shrk < r:
            r_target = max(r_shrk, max(self.cfg.r_min, 1))
            r_new = max(r - self.cfg.shrink_step, r_target)
            if r_new != r:
                if callable(self.on_rank_change): self.on_rank_change(r, r_new)
                self.rank_active.fill_(int(r_new))
                self._last_adapt_step.copy_(self._step)
            return

    @torch.no_grad()
    def current_rank(self) -> int:
        return int(self.rank_active.item())

    @torch.no_grad()
    def adapt_now(self):
        if self._should_track() and int(self._step.item()) >= self.cfg.warmup_steps:
            self._maybe_adapt_rank()


# -------- Wrapping / materialize / utils --------

def _get_parent(root: nn.Module, name: str) -> Optional[nn.Module]:
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return None
        parent = getattr(parent, p)
    return parent

def iter_adaptive_layers(model: nn.Module) -> Iterable[nn.Module]:
    for m in model.modules():
        if isinstance(m, (AdaptiveLinear, AdaptiveConv2d)):
            yield m

def set_rank_change_callback(model: nn.Module, callback: Callable[[int, int], None]):
    for m in model.modules():
        if isinstance(m, (AdaptiveLinear, AdaptiveConv2d)):
            m.on_rank_change = callback

def wrap_model_with_adaptive(
    model: nn.Module,
    eps: float = 0.08,
    r_min: int = 4,
    r_max: int = 128,
    sketch_size: int = 256,
    adapt_every: int = 50,
    grow_step: int = 16,
    shrink_step: int = 32,
    track_inputs_every: int = 1,
    include_conv: bool = True,
    freeze_during_eval: bool = True,
    sample_stride: int = 1,
    max_rows_fd: int = 2048,
    warmup_steps: int = 0,
    svd_backend: str = "auto",
    persist_sketch: Optional[bool] = None,
    min_rows_fd: Optional[int] = None,
    adapt_cooldown: Optional[int] = None,
    disable_amp_for_fd: Optional[bool] = None,
    eps_grow_override: Optional[float] = None,
    eps_shrink_override: Optional[float] = None,
) -> nn.Module:
    # Fenêtre d'hystérésis par défaut
    eps_grow = eps_grow_override if eps_grow_override is not None else max(1e-6, eps * 0.875)
    eps_shrink = eps_shrink_override if eps_shrink_override is not None else max(eps_grow + 1e-6, eps * 1.125)

    cfg = AdaptConfig(
        eps_grow=eps_grow,
        eps_shrink=eps_shrink,
        r_min=r_min,
        r_max=r_max,
        sketch_size=sketch_size,
        adapt_every=adapt_every,
        grow_step=grow_step,
        shrink_step=shrink_step,
        track_inputs_every=track_inputs_every,
        freeze_during_eval=freeze_during_eval,
        sample_stride=sample_stride,
        max_rows_fd=max_rows_fd,
        warmup_steps=warmup_steps,
        svd_backend=svd_backend,
    )
    if persist_sketch is not None: cfg.persist_sketch = bool(persist_sketch)
    if min_rows_fd is not None: cfg.min_rows_fd = int(min_rows_fd)
    if adapt_cooldown is not None: cfg.adapt_cooldown = int(adapt_cooldown)
    if disable_amp_for_fd is not None: cfg.disable_amp_for_fd = bool(disable_amp_for_fd)

    for name, m in list(model.named_modules()):
        if name == "":  # root
            continue
        parent = _get_parent(model, name)
        if parent is None:
            continue
        child_name = name.split(".")[-1]
        leaf = len(list(m.children())) == 0
        if isinstance(m, (AdaptiveLinear, AdaptiveConv2d)):
            continue
        if leaf and isinstance(m, nn.Linear):
            setattr(parent, child_name, AdaptiveLinear(m, cfg))
        elif leaf and include_conv and isinstance(m, nn.Conv2d):
            if m.groups != 1:
                warnings.warn(f"[wrap] Conv2d '{name}' non remplacée (groups={m.groups}).", RuntimeWarning)
            else:
                setattr(parent, child_name, AdaptiveConv2d(m, cfg))
    return model

@torch.no_grad()
def materialize_linear(m: AdaptiveLinear) -> nn.Linear:
    r = int(m.rank_active.item())
    W = (m.A_full.weight[:, :r] @ m.B_full.weight[:r, :]).to(m.A_full.weight.dtype)
    out_f, in_f = W.shape
    lin = nn.Linear(in_f, out_f, bias=True, device=W.device, dtype=W.dtype)
    lin.weight.copy_(W)
    lin.bias.copy_(m.A_full.bias)
    return lin

@torch.no_grad()
def materialize_conv2d(m: AdaptiveConv2d) -> nn.Conv2d:
    r = int(m.rank_active.item())
    W1 = m.conv1_full.weight[:r, :, :, :]
    W2 = m.conv2_full.weight[:, :r, :, :]
    cout, _, _, _ = W2.shape
    r_, cin, kH, kW = W1.shape
    assert r_ == r
    W_eff = torch.einsum("or,rcij->ocij", W2.view(cout, r), W1).contiguous()
    conv = nn.Conv2d(
        in_channels=cin, out_channels=cout, kernel_size=(kH, kW),
        stride=m.stride, padding=m.padding, dilation=m.dilation,
        bias=True, device=W_eff.device, dtype=W_eff.dtype,
    )
    conv.weight.copy_(W_eff.to(conv.weight.dtype))
    conv.bias.copy_(m.conv2_full.bias.to(conv.bias.dtype))
    return conv

def materialize_model(model: nn.Module) -> nn.Module:
    import copy
    model_out = copy.deepcopy(model)
    for name, m in list(model_out.named_modules()):
        if name == "": continue
        parent = _get_parent(model_out, name)
        if parent is None: continue
        child_name = name.split(".")[-1]
        if isinstance(m, AdaptiveLinear):
            setattr(parent, child_name, materialize_linear(m))
        elif isinstance(m, AdaptiveConv2d):
            setattr(parent, child_name, materialize_conv2d(m))
    return model_out

@torch.no_grad()
def clear_fd_sketch(model: nn.Module):
    for m in model.modules():
        if isinstance(m, (AdaptiveLinear, AdaptiveConv2d)):
            if hasattr(m, "fd"):
                m.fd.B.zero_()
                m.fd.n_seen = 0
                m._sync_fd_to_buffers()

# ===============================
# Bench & Serving
# ===============================

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def build_model(name: str, num_classes: int = 10) -> nn.Module:
    """
    name options:
      - torchvision: 'resnet18' (par défaut)
      - timm: prefix 'timm_' puis nom timm (ex: 'timm_resnet50', 'timm_vit_tiny_patch16_224')
    """
    if name.startswith("timm_"):
        try:
            import timm
        except Exception as e:
            raise RuntimeError("timm n'est pas installé. pip install timm") from e
        timm_name = name[len("timm_"):]
        # La plupart des modèles timm acceptent num_classes directement.
        # Pour les modèles qui nécessitent un head spécifique, timm gère l'injection.
        return timm.create_model(timm_name, pretrained=False, num_classes=num_classes)
    else:
        from torchvision.models import resnet18
        return resnet18(num_classes=num_classes)

def get_data(load_real: bool, batch_size: int, img_size: int = 224):
    """
    - offline (--no-data): données synthétiques.
    - en ligne: CIFAR-10 + normalisation (moyenne/écart-type CIFAR-10).
    """
    if not load_real:
        def _rand_loader(n_batches=50):
            for _ in range(n_batches):
                yield torch.randn(batch_size, 3, img_size, img_size), torch.randint(0, 10, (batch_size,))
        return _rand_loader(), _rand_loader()

    # CIFAR-10 (normalisation)
    from torchvision import datasets, transforms
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std  = (0.2470, 0.2435, 0.2616)
    transform_train = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std),
    ])
    train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform_train)
    test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    pin = torch.cuda.is_available()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=pin)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=pin)
    return train_loader, test_loader

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += int((pred == y).sum().item())
        total += int(y.numel())
    return (correct / max(1, total)) * 100.0

def train_one_epoch(model: nn.Module, loader, device: torch.device, lr=5e-4, amp=True):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(amp and device.type == "cuda"))
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

def measure_latency(model: nn.Module, device: torch.device, img_size=224, batch=128, warmup=10, iters=50) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch, 3, img_size, img_size, device=device)
    with torch.no_grad():
        # warmup
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
    total = t1 - t0
    lat = total / max(1, iters)
    thr = (batch * iters) / max(1e-9, total)
    return {"latency_s": float(lat), "throughput_samp_s": float(thr)}

def vram_reset_peak():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

def vram_read_peak_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    peak = float(torch.cuda.max_memory_allocated()) / (1024**3)
    return peak

def maybe_compile(model: nn.Module, use_compile: bool):
    if not use_compile:
        return model
    try:
        model = torch.compile(model)  # PyTorch 2.x
        print("[compile] torch.compile activé")
        return model
    except Exception as e:
        warnings.warn(f"torch.compile indisponible/échec : {e}")
        return model

def export_artifacts(dense: nn.Module, device: torch.device, img_size: int, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    dense.eval().to(device)
    # TorchScript
    try:
        scripted = torch.jit.script(dense)
        scripted.save(os.path.join(outdir, "model_scripted.pt"))
        print("[export] TorchScript OK")
    except Exception as e:
        warnings.warn(f"[export] TorchScript échec: {e}")
    # ONNX
    try:
        x = torch.randn(1, 3, img_size, img_size, device=device)
        torch.onnx.export(
            dense, x, os.path.join(outdir, "model.onnx"),
            opset_version=17, input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}}
        )
        print("[export] ONNX OK")
    except Exception as e:
        warnings.warn(f"[export] ONNX échec (install onnx?): {e}")

def log_adapt_stats(model: nn.Module) -> Dict[str, Any]:
    ranks = []
    totals = []
    for m in iter_adaptive_layers(model):
        s = m.get_rank_stats()
        ranks.append(s["rank_active"])
        totals.append(s.get("energy_total", 0.0))
    out = {
        "num_adaptive_layers": len(ranks),
        "rank_mean": float(sum(ranks)/max(1,len(ranks))) if ranks else 0.0,
        "ranks": ranks,
        "fd_energy_totals": totals,
    }
    return out

def serve_fastapi(dense: nn.Module, device: torch.device, host="0.0.0.0", port=8000, img_size=224):
    try:
        from fastapi import FastAPI
        import uvicorn
    except Exception as e:
        raise RuntimeError("FastAPI/uvicorn non installés. pip install fastapi uvicorn") from e

    app = FastAPI(title="Green Serving (materialized)")
    dense.eval().to(device)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    def predict():
        # Démo : input synthétique (pour garder monolithique).
        x = torch.randn(1, 3, img_size, img_size, device=device)
        with torch.no_grad():
            y = dense(x).softmax(dim=1)
        probs = y[0].tolist()
        return {"probs": probs[:10]}

    uvicorn.run(app, host=host, port=port)

# ===============================
# CLI principal
# ===============================

def parse_args():
    ap = argparse.ArgumentParser(description="Green Serving Demo – AdaptiveLowRank")
    ap.add_argument("--model", type=str, default="resnet18",
                    help="resnet18 | timm_<model> (ex: timm_resnet50, timm_vit_tiny_patch16_224)")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--eps", type=float, default=0.08)
    ap.add_argument("--eps-grow", type=float, default=None, help="Override précis pour eps_grow (sinon dérivé de --eps)")
    ap.add_argument("--eps-shrink", type=float, default=None, help="Override précis pour eps_shrink (sinon dérivé de --eps)")
    ap.add_argument("--r-min", type=int, default=4)
    ap.add_argument("--r-max", type=int, default=128)
    ap.add_argument("--sketch-size", type=int, default=256)
    ap.add_argument("--adapt-every", type=int, default=50)
    ap.add_argument("--grow-step", type=int, default=16)
    ap.add_argument("--shrink-step", type=int, default=32)
    ap.add_argument("--warmup-steps", type=int, default=0)
    ap.add_argument("--max-rows-fd", type=int, default=2048)
    ap.add_argument("--sample-stride", type=int, default=1)
    ap.add_argument("--cooldown", type=int, default=10)
    ap.add_argument("--no-data", action="store_true", help="utilise des données synthétiques (pas de CIFAR-10)")
    ap.add_argument("--compile", action="store_true", help="torch.compile si dispo")
    ap.add_argument("--serve", action="store_true", help="lancer un serveur FastAPI sur le modèle matérialisé")
    ap.add_argument("--save-out", type=str, default="artifacts", help="dossier d'exports")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    device = get_device()
    print(f"[device] {device}")

    # 1) Modèle de base
    base = build_model(args.model, num_classes=args.num_classes).to(device)
    base = maybe_compile(base, args.compile)

    # 2) Wrap adaptatif
    model = wrap_model_with_adaptive(
        base,
        eps=args.eps,
        r_min=args.r_min,
        r_max=args.r_max,
        sketch_size=args.sketch_size,
        adapt_every=args.adapt_every,
        grow_step=args.grow_step,
        shrink_step=args.shrink_step,
        track_inputs_every=1,
        include_conv=True,
        freeze_during_eval=True,
        sample_stride=args.sample_stride,
        max_rows_fd=args.max_rows_fd,
        warmup_steps=args.warmup_steps,
        svd_backend="auto",
        adapt_cooldown=args.cooldown,
        eps_grow_override=args.eps_grow,
        eps_shrink_override=args.eps_shrink,
    ).to(device)

    def _log_rank_change(old, new):
        print(f"[rank-change] {old} → {new}")
    set_rank_change_callback(model, _log_rank_change)

    # 3) Données
    train_loader, test_loader = get_data(load_real=(not args.no_data), batch_size=args.batch, img_size=args.img_size)

    # 4) Avant entraînement: quick eval + latence + VRAM (mesure corrigée)
    pre_eval = evaluate(model, test_loader, device)
    print(f"[pre] accuracy={pre_eval:.2f}%")

    vram_reset_peak()
    pre_lat = measure_latency(model, device, img_size=args.img_size, batch=args.batch)
    pre_vram = vram_read_peak_gb()
    print(f"[pre] latency={pre_lat['latency_s']:.4f}s  thr={pre_lat['throughput_samp_s']:.1f}/s  VRAM_peak={pre_vram:.2f} GB")
    print(f"[pre] adapt stats: {log_adapt_stats(model)}")

    # 5) Entraînement bref (adaptation ON)
    use_amp = (device.type == "cuda") and (not args.deterministic)  # pour cohérence avec determinism
    for ep in range(args.epochs):
        print(f"== Epoch {ep} ==")
        train_one_epoch(model, train_loader, device, amp=use_amp)
        acc = evaluate(model, test_loader, device)
        print(f"[epoch {ep}] acc={acc:.2f}%  stats={log_adapt_stats(model)}")

    # 6) Latence post-entrainement
    vram_reset_peak()
    post_lat = measure_latency(model, device, img_size=args.img_size, batch=args.batch)
    post_vram = vram_read_peak_gb()
    print(f"[post] latency={post_lat['latency_s']:.4f}s  thr={post_lat['throughput_samp_s']:.1f}/s  VRAM_peak={post_vram:.2f} GB")

    # 7) Matérialisation → modèle dense fixe (pour servir)
    dense = materialize_model(model).to(device)
    dense.eval()
    mat_acc = evaluate(dense, test_loader, device)

    vram_reset_peak()
    mat_lat = measure_latency(dense, device, img_size=args.img_size, batch=args.batch)
    mat_vram = vram_read_peak_gb()
    print(f"[materialized] acc={mat_acc:.2f}%  latency={mat_lat['latency_s']:.4f}s  thr={mat_lat['throughput_samp_s']:.1f}/s  VRAM_peak={mat_vram:.2f} GB")

    # 8) Exports
    os.makedirs(args.save_out, exist_ok=True)
    torch.save({"model_adaptive": model.state_dict()}, os.path.join(args.save_out, "adaptive_state_dict.pt"))
    torch.save({"model_dense": dense.state_dict()}, os.path.join(args.save_out, "dense_state_dict.pt"))
    export_artifacts(dense, device, args.img_size, args.save_out)

    # 9) Résumé CSV minimal
    csv_path = os.path.join(args.save_out, "bench_summary.csv")
    with open(csv_path, "w") as f:
        f.write("phase,latency_s,throughput_samp_s,VRAM_GB\n")
        f.write(f"pre,{pre_lat['latency_s']:.6f},{pre_lat['throughput_samp_s']:.3f},{pre_vram:.3f}\n")
        f.write(f"post,{post_lat['latency_s']:.6f},{post_lat['throughput_samp_s']:.3f},{post_vram:.3f}\n")
        f.write(f"materialized,{mat_lat['latency_s']:.6f},{mat_lat['throughput_samp_s']:.3f},{mat_vram:.3f}\n")
    print(f"[out] Résumé écrit: {csv_path}")

    # 10) Option service FastAPI
    if args.serve:
        print("[serve] lancement FastAPI avec le modèle matérialisé…")
        serve_fastapi(dense, device, host="0.0.0.0", port=8000, img_size=args.img_size)

if __name__ == "__main__":
    main()
