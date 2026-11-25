"""Small neural forecasters for the declared extension (docs/PREANALYSIS.md, Addendum A).

* ``mlp``  : feed-forward network on the same seven history summaries as the core families.
* ``tcn``  : causal dilated temporal convolution on the last ``seq_len`` daily returns.
* ``attn`` : one-layer transformer encoder on the same return sequence.

All three are trained like the core families: QLIKE with a log link, so the level target is
E[Y | inputs]; an ensemble of members trained jointly, with the forecast the mean of the members
on the level scale; epochs chosen on the purged validation block, then a refit on all eligible
rows.  Members are stacked on a leading axis, so one optimiser step trains the whole ensemble.

Training runs in PyTorch, on a GPU when one is visible.  The forward pass is written once
against a tiny array interface and is evaluated with NumPy at prediction time.  A fitted model
therefore holds only NumPy arrays: it pickles cleanly, and the downstream evaluation (projection,
matching, placebos) runs on CPU without CUDA contexts.  ``tests/test_neural.py`` checks that the
two backends agree.
"""
from __future__ import annotations

import os

import numpy as np

from .data import HIST_COLS, VARIANCE_COLS
from .models import _Base, qlike

N_HIST = len(HIST_COLS)
_VAR_IDX = [HIST_COLS.index(c) for c in VARIANCE_COLS]
_LOG_EPS = 1e-7
_ETA_CLIP = 10.0          # on the log scale of y / mean(y_train): far beyond any observed ratio
_TCN_LEVELS = 5           # dilations 1..16: receptive field 63 sessions
_N_HEADS = 2


# ------------------------------------------------------------------------------------
# One forward pass, two backends
# ------------------------------------------------------------------------------------
class NumpyOps:
    matmul = staticmethod(np.matmul)
    exp = staticmethod(np.exp)

    @staticmethod
    def relu(x):
        return np.maximum(x, 0.0)

    @staticmethod
    def cat(xs):
        return np.concatenate(xs, axis=-1)

    @staticmethod
    def pad_time(x, n):
        return np.pad(x, [(0, 0)] * (x.ndim - 2) + [(n, 0), (0, 0)])

    @staticmethod
    def softmax(x):
        z = np.exp(x - x.max(axis=-1, keepdims=True))
        return z / z.sum(axis=-1, keepdims=True)

    @staticmethod
    def expand(v, n_members):
        return np.broadcast_to(v[None], (n_members,) + v.shape)

    @staticmethod
    def mean(x, axis, keepdims=False):
        return x.mean(axis=axis, keepdims=keepdims)

    @staticmethod
    def swap(x, a, b):
        return np.swapaxes(x, a, b)


class TorchOps:
    def __init__(self, torch):
        self.t = torch

    def matmul(self, a, b):
        return self.t.matmul(a, b)

    def exp(self, x):
        return self.t.exp(x)

    def relu(self, x):
        return self.t.relu(x)

    def cat(self, xs):
        return self.t.cat(xs, dim=-1)

    def pad_time(self, x, n):
        return self.t.nn.functional.pad(x, (0, 0, n, 0))

    def softmax(self, x):
        return self.t.softmax(x, dim=-1)

    def expand(self, v, n_members):
        return v.unsqueeze(0).expand(n_members, *v.shape)

    def mean(self, x, axis, keepdims=False):
        return x.mean(dim=axis, keepdim=keepdims)

    def swap(self, x, a, b):
        return x.transpose(a, b)

    def tcn(self, p, x):
        """Same network as ``_conv`` levels, as grouped cuDNN convolutions (one group per member).

        The matmul form reduces over N*L rows into a 16x16 weight gradient, which cuBLAS runs on a
        handful of blocks; the convolution form trains about 6x faster.  Output (E, N, H) at the
        last session, equal to the NumPy path up to float rounding (``tests/test_neural.py``).
        """
        F = self.t.nn.functional
        E = p["Wout"].shape[0]
        N = x.shape[0]
        h = x.transpose(1, 2)                                    # (N, C, L)
        for k in range(_TCN_LEVELS):
            W, b = p[f"W{k}"], p[f"b{k}"]
            _, c_in, H, _ = W.shape
            w = W.permute(0, 2, 1, 3).reshape(E * H, c_in, 3)  # tap j reads x[t - (2 - j) d]
            d = 2**k
            y = self.relu(F.conv1d(F.pad(h, (2 * d, 0)), w, b.reshape(E * H), dilation=d,
                                   groups=1 if k == 0 else E))
            h = y if k == 0 else h + y
        return h[:, :, -1].reshape(N, E, -1).transpose(0, 1)


def _layer_norm(ops, x, g, b):
    mu = ops.mean(x, -1, True)
    var = ops.mean((x - mu) ** 2, -1, True)
    return (x - mu) / (var + 1e-5) ** 0.5 * g + b


def _conv(ops, x, W, b, dilation, first):
    """Causal dilated convolution, kernel 3, ensemble-batched.  x: (N, L, C) or (E, N, L, C)."""
    E, C, H, _ = W.shape
    L = x.shape[-2]
    N = x.shape[0] if first else x.shape[1]
    xp = ops.pad_time(x, 2 * dilation)
    out = 0.0
    for j in range(3):             # tap j reads x[t - (2 - j) * dilation]
        xs = xp[..., j * dilation : j * dilation + L, :]
        xs = xs.reshape(1 if first else E, N * L, C)
        out = out + ops.matmul(xs, W[..., j]).reshape(E, N, L, H)
    return out + b


def forward(p: dict, inp: dict, arch: str, ops) -> object:
    """Log-scale outputs eta with shape (members, rows)."""
    E = p["Wout"].shape[0]
    if arch == "mlp":
        h = ops.relu(ops.matmul(inp["tab"], p["W0"]) + p["b0"])
        k = 1
        while f"W{k}" in p:
            h = ops.relu(ops.matmul(h, p[f"W{k}"]) + p[f"b{k}"])
            k += 1
        return (ops.matmul(h, p["Wout"]) + p["bout"])[..., 0]

    x = inp["seq"]
    N, L, C = x.shape
    if arch == "tcn" and hasattr(ops, "tcn"):
        z = ops.tcn(p, x)
    elif arch == "tcn":
        for k in range(_TCN_LEVELS):
            y = ops.relu(_conv(ops, x, p[f"W{k}"], p[f"b{k}"], 2**k, first=k == 0))
            x = y if k == 0 else x + y
        z = x[:, :, -1, :]
    elif arch == "attn":
        d = p["We"].shape[-1]
        dk = d // _N_HEADS
        h = ops.matmul(x.reshape(1, N * L, C), p["We"]).reshape(E, N, L, d) + p["be"] + p["pos"]
        a = _layer_norm(ops, h, p["g1"], p["c1"]).reshape(E, N * L, d)

        def heads(W):
            return ops.swap(ops.matmul(a, W).reshape(E, N, L, _N_HEADS, dk), 2, 3)

        q, k_, v = heads(p["Wq"]), heads(p["Wk"]), heads(p["Wv"])
        att = ops.matmul(ops.softmax(ops.matmul(q, ops.swap(k_, -1, -2)) / dk**0.5), v)
        att = ops.swap(att, 2, 3).reshape(E, N * L, d)
        h = h + ops.matmul(att, p["Wo"]).reshape(E, N, L, d)
        a = _layer_norm(ops, h, p["g2"], p["c2"]).reshape(E, N * L, d)
        f = ops.relu(ops.matmul(a, p["F1"]).reshape(E, N, L, 2 * d) + p["f1"])
        h = h + ops.matmul(f.reshape(E, N * L, 2 * d), p["F2"]).reshape(E, N, L, d) + p["f2"]
        z = ops.mean(h, 2)
    else:
        raise ValueError(arch)
    if inp.get("v") is not None:
        z = ops.cat([z, ops.expand(inp["v"], E)])
    z = ops.relu(ops.matmul(z, p["Wh"]) + p["bh"])
    return (ops.matmul(z, p["Wout"]) + p["bout"])[..., 0]


# ------------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------------
def init_params(arch: str, dims: dict, hp: dict, n_members: int, gen, torch) -> dict:
    E = n_members

    def w(*shape, fan_in, scale=1.0):
        return torch.randn(*shape, generator=gen) * (scale * (2.0 / fan_in) ** 0.5)

    def zeros(*shape):
        return torch.zeros(*shape)

    p = {}
    nv = dims["v"]
    if arch == "mlp":
        H, size = hp["hidden"], dims["tab"]
        p["W0"], p["b0"] = w(E, size, H, fan_in=size), zeros(E, 1, H)
        for k in range(1, hp["depth"]):
            p[f"W{k}"], p[f"b{k}"] = w(E, H, H, fan_in=H), zeros(E, 1, H)
        last = H
    else:
        C, L = dims["channels"], dims["seq_len"]
        if arch == "tcn":
            H = hp["hidden"]
            p["W0"], p["b0"] = w(E, C, H, 3, fan_in=3 * C), zeros(E, 1, 1, H)
            for k in range(1, _TCN_LEVELS):
                p[f"W{k}"], p[f"b{k}"] = w(E, H, H, 3, fan_in=3 * H, scale=0.5), zeros(E, 1, 1, H)
        else:
            H = d = hp["d_model"]
            p["We"], p["be"] = w(E, C, d, fan_in=C), zeros(E, 1, 1, d)
            p["pos"] = 0.02 * torch.randn(E, 1, L, d, generator=gen)
            for name in ("Wq", "Wk", "Wv", "Wo"):
                p[name] = w(E, d, d, fan_in=d, scale=0.5)
            p["g1"], p["c1"] = torch.ones(E, 1, 1, d), zeros(E, 1, 1, d)
            p["g2"], p["c2"] = torch.ones(E, 1, 1, d), zeros(E, 1, 1, d)
            p["F1"], p["f1"] = w(E, d, 2 * d, fan_in=d), zeros(E, 1, 1, 2 * d)
            p["F2"], p["f2"] = w(E, 2 * d, d, fan_in=2 * d, scale=0.5), zeros(E, 1, 1, d)
        p["Wh"], p["bh"] = w(E, H + nv, H, fan_in=H + nv), zeros(E, 1, H)
        last = H
    p["Wout"], p["bout"] = w(E, last, 1, fan_in=last, scale=0.1), zeros(E, 1, 1)
    return p


def _decayed(name: str) -> bool:
    """Weight matrices and positional embeddings decay; biases and norm gains do not."""
    return name.startswith(("W", "F")) or name == "pos"


_PROCESS_GPU: int | None = None


def _next_gpu(n_gpus: int) -> int:
    """Round-robin over GPUs across all worker processes of a run, through a locked counter
    file.  PID parity, the earlier rule, can put most workers of a pool on one card."""
    import fcntl
    import tempfile

    path = os.path.join(os.environ.get("TMPDIR", tempfile.gettempdir()), "ivbehav_gpu_counter")
    with open(path, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        fh.seek(0)
        text = fh.read().strip()
        k = int(text) if text.isdigit() else 0
        fh.seek(0)
        fh.truncate()
        fh.write(str(k + 1))
        fh.flush()
    return k % n_gpus


def _device(torch):
    global _PROCESS_GPU
    want = os.environ.get("IVBEHAV_DEVICE", "auto")
    if want == "cpu" or not torch.cuda.is_available():
        torch.set_num_threads(1)
        return torch.device("cpu")
    if want.startswith("cuda:"):
        return torch.device(want)
    if _PROCESS_GPU is None:          # one card per worker process, kept for its lifetime
        _PROCESS_GPU = _next_gpu(torch.cuda.device_count())
    return torch.device(f"cuda:{_PROCESS_GPU}")


def train_ensemble(arch, prep, y, hp, nn_cfg, seed, ybar, floor, epochs=None, prep_val=None, y_val=None):
    """Full-batch AdamW on mean QLIKE.  With validation data, returns the epoch whose
    ensemble-mean forecast scores best (evaluated every ``eval_every`` epochs).

    A CUDA out-of-memory error (another job on the card, or too many workers) does not stop the
    run: the fit is retried from the same initialisation on the other card after a pause, and
    on CPU as a last resort.  Cached GPU memory is released after every fit, so idle workers do
    not hold it.
    """
    import time

    import torch

    devices = [_device(torch)]
    if devices[0].type == "cuda":
        n = torch.cuda.device_count()
        devices += [torch.device(f"cuda:{(devices[0].index + i) % n}") for i in range(1, n)]
        devices += [devices[0], torch.device("cpu")]
    for attempt, dev in enumerate(devices):
        try:
            return _train_on(dev, arch, prep, y, hp, nn_cfg, seed, ybar, floor, epochs, prep_val, y_val)
        except torch.cuda.OutOfMemoryError:
            if attempt == len(devices) - 1:
                raise
        finally:                      # runs after the failed fit's tensors are released
            if dev.type == "cuda":
                torch.cuda.empty_cache()
        print(f"[neural] CUDA out of memory on {dev} ({arch}); retrying on {devices[attempt + 1]}", flush=True)
        time.sleep(30 * (attempt + 1))


def _train_on(dev, arch, prep, y, hp, nn_cfg, seed, ybar, floor, epochs, prep_val, y_val):
    import torch

    if dev.type == "cpu":
        torch.set_num_threads(1)
    gen = torch.Generator().manual_seed(int(seed))
    dims = {"tab": prep["tab"].shape[1] if prep.get("tab") is not None else 0,
            "v": 0 if prep.get("v") is None else 1,
            "channels": prep["seq"].shape[2] if prep.get("seq") is not None else 0,
            "seq_len": prep["seq"].shape[1] if prep.get("seq") is not None else 0}
    params = {k: v.to(dev).requires_grad_() for k, v in
              init_params(arch, dims, hp, nn_cfg["n_members"], gen, torch).items()}
    ops = TorchOps(torch)

    def tensors(d):
        return {k: torch.as_tensor(v, device=dev) for k, v in d.items() if v is not None}

    data = tensors(prep)
    target = torch.as_tensor((y / ybar).astype(np.float32), device=dev)
    opt = torch.optim.AdamW(
        [{"params": [v for k, v in params.items() if _decayed(k)], "weight_decay": hp["weight_decay"]},
         {"params": [v for k, v in params.items() if not _decayed(k)], "weight_decay": 0.0}],
        lr=hp["lr"])

    validate = prep_val is not None
    val = tensors(prep_val) if validate else None
    n_epochs = int(epochs) if epochs is not None else int(nn_cfg["max_epochs"])
    every, patience = int(nn_cfg["eval_every"]), int(nn_cfg["patience"])
    best_score, best_epoch = np.inf, every
    for epoch in range(1, n_epochs + 1):
        eta = forward(params, data, arch, ops).clamp(-_ETA_CLIP, _ETA_CLIP)
        loss = (target * torch.exp(-eta) + eta).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if validate and epoch % every == 0:
            with torch.no_grad():
                eta_v = forward(params, val, arch, ops).clamp(-_ETA_CLIP, _ETA_CLIP)
                mu = ybar * torch.exp(eta_v.double()).mean(dim=0).cpu().numpy()
            score = float(np.mean(qlike(y_val, np.maximum(mu, floor))))
            if score < best_score - 1e-9:
                best_score, best_epoch = score, epoch
            elif epoch - best_epoch >= patience:
                break
    out = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in params.items()}
    return out, best_epoch, best_score


# ------------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------------
class NeuralModel(_Base):
    """f0 or f1 for one neural architecture; ``hp['epochs']`` fixes the training length."""

    def __init__(self, arch: str, hp: dict, nn_cfg: dict):
        self.arch, self.hp, self.nn = arch, dict(hp), dict(nn_cfg)

    # Training-only transforms ---------------------------------------------------------
    def _fit_transforms(self, X, V, y):
        self.ybar_ = float(y.mean())
        if self.arch == "mlp":
            T = self._log_hist(X)
            self.tm_, self.ts_ = T.mean(0), np.maximum(T.std(0), 1e-12)
        else:
            R = self._returns(X)
            if R.shape[1] == 0:
                raise ValueError("sequence models need seq_len return columns after the 7 summaries")
            self.rs_ = float(R.std())
            lr2 = np.log(R**2 + _LOG_EPS)
            self.lm_, self.ls_ = float(lr2.mean()), float(lr2.std())
        if V is not None:
            lv = np.log(V)
            self.vm_, self.vs_ = float(lv.mean()), float(lv.std())

    @staticmethod
    def _log_hist(X):
        T = np.asarray(X, float)[:, :N_HIST].copy()
        T[:, _VAR_IDX] = np.log(np.maximum(T[:, _VAR_IDX], 0.0) + _LOG_EPS)
        return T

    @staticmethod
    def _returns(X):
        return np.asarray(X, float)[:, N_HIST:]

    def _prepare(self, X, V) -> dict:
        v = None if V is None else ((np.log(V) - self.vm_) / self.vs_)[:, None].astype(np.float32)
        if self.arch == "mlp":
            tab = (self._log_hist(X) - self.tm_) / self.ts_
            if v is not None:
                tab = np.column_stack([tab, v])
            return {"tab": tab.astype(np.float32)}
        R = self._returns(X)
        seq = np.stack([R / self.rs_, (np.log(R**2 + _LOG_EPS) - self.lm_) / self.ls_], axis=-1)
        return {"seq": seq.astype(np.float32), "v": v}

    # Fitting ----------------------------------------------------------------------------
    def fit(self, X, V, y, floor, seed=0, X_val=None, V_val=None, y_val=None, **_):
        self.uses_v, self.floor = V is not None, floor
        self._fit_transforms(X, V, y)
        prep = self._prepare(X, V)
        if y_val is not None:
            params, epochs, score = train_ensemble(
                self.arch, prep, y, self.hp, self.nn, seed, self.ybar_, floor,
                prep_val=self._prepare(X_val, V_val), y_val=y_val)
            self.best_epochs_, self.best_val_ = epochs, score
        else:
            params, _, _ = train_ensemble(self.arch, prep, y, self.hp, self.nn, seed, self.ybar_, floor,
                                          epochs=self.hp["epochs"])
        self.params_ = params
        return self

    def _raw(self, X, V):
        prep = self._prepare(X, V)
        n = len(X)
        chunk = int(self.nn.get("predict_chunk", 512))
        out = np.empty(n)
        for start in range(0, n, chunk):
            sub = {k: (None if v is None else v[start : start + chunk]) for k, v in prep.items()}
            eta = np.clip(forward(self.params_, sub, self.arch, NumpyOps), -_ETA_CLIP, _ETA_CLIP)
            out[start : start + chunk] = self.ybar_ * np.exp(eta.astype(np.float64)).mean(axis=0)
        return out
