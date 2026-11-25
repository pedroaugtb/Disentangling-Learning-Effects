"""Neural extension: backend agreement, level targeting, pickling, and isolation of return lags."""
import dataclasses
import os
import pickle

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
os.environ.setdefault("IVBEHAV_DEVICE", "cpu")

from ivbehav.conditional import LawBasis  # noqa: E402
from ivbehav.data import HIST_COLS, return_lags  # noqa: E402
from ivbehav.folds import make_folds  # noqa: E402
from ivbehav.models import fit_pair, make_model, validation_table  # noqa: E402
from ivbehav.neural import NeuralModel, NumpyOps, TorchOps, forward, init_params  # noqa: E402

NN = {"n_members": 4, "max_epochs": 200, "eval_every": 25, "patience": 50, "predict_chunk": 64}
HP = {"mlp": {"hidden": 16, "depth": 2, "lr": 3e-3, "weight_decay": 1e-3},
      "tcn": {"hidden": 8, "lr": 3e-3, "weight_decay": 1e-3},
      "attn": {"d_model": 8, "lr": 3e-3, "weight_decay": 1e-3}}


def _with_lags(data, seq_len=66):
    lags = return_lags(pd.Series(data.ret), seq_len)
    return dataclasses.replace(data, X=np.column_stack([data.X, lags]))


@pytest.mark.parametrize("arch", ["mlp", "tcn", "attn"])
@pytest.mark.parametrize("with_v", [False, True])
def test_numpy_and_torch_forward_agree(arch, with_v):
    rng = np.random.default_rng(0)
    gen = torch.Generator().manual_seed(1)
    n, L = 37, 66
    inp = {"tab": rng.standard_normal((n, 8 if with_v else 7)).astype(np.float32),
           "seq": rng.standard_normal((n, L, 2)).astype(np.float32),
           "v": rng.standard_normal((n, 1)).astype(np.float32) if with_v else None}
    dims = {"tab": inp["tab"].shape[1], "v": int(with_v), "channels": 2, "seq_len": L}
    params = init_params(arch, dims, HP[arch], 3, gen, torch)
    ref = forward(params, {k: (None if v is None else torch.as_tensor(v)) for k, v in inp.items()},
                  arch, TorchOps(torch)).detach().numpy()
    got = forward({k: v.numpy() for k, v in params.items()}, inp, arch, NumpyOps)
    assert got.shape == (3, n)
    np.testing.assert_allclose(got, ref, rtol=1e-4, atol=1e-5)


def test_tcn_is_causal():
    """Changing the oldest returns outside the receptive field leaves the output unchanged;
    changing the newest return changes it."""
    gen = torch.Generator().manual_seed(2)
    rng = np.random.default_rng(3)
    params = {k: v.numpy() for k, v in init_params("tcn", {"tab": 0, "v": 0, "channels": 2, "seq_len": 66},
                                                    HP["tcn"], 2, gen, torch).items()}
    seq = rng.standard_normal((5, 66, 2)).astype(np.float32)
    base = forward(params, {"seq": seq}, "tcn", NumpyOps)
    old = seq.copy()
    old[:, :2, :] += 5.0                 # receptive field is 63 sessions
    np.testing.assert_allclose(forward(params, {"seq": old}, "tcn", NumpyOps), base, atol=1e-6)
    new = seq.copy()
    new[:, -1, :] += 1.0
    assert np.abs(forward(params, {"seq": new}, "tcn", NumpyOps) - base).max() > 1e-4


def test_return_lags_are_predetermined():
    r = pd.Series(np.arange(10, dtype=float))
    lags = return_lags(r, 3)
    np.testing.assert_array_equal(lags[5], [3.0, 4.0, 5.0])     # r_{t-2}, r_{t-1}, r_t
    assert np.isnan(lags[1, 0])


def test_summary_modules_ignore_return_lags(synth):
    ext = _with_lags(synth)
    rows = np.flatnonzero(np.isfinite(ext.X).all(axis=1))[:500]
    np.testing.assert_allclose(LawBasis().fit(ext.X[rows]).z(ext.X[rows]),
                               LawBasis().fit(synth.X[rows]).z(synth.X[rows]))


@pytest.mark.parametrize("arch", ["mlp", "tcn", "attn"])
def test_training_targets_the_level_and_pickles(arch, synth, cfg):
    data = _with_lags(synth)
    local = {**cfg, "neural": NN}
    fold = make_folds(data, local, local["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    hp = {**HP[arch], "epochs": 150}
    m0, m1 = fit_pair(arch, data.X[tr], data.V[tr], data.y[tr], hp, hp, local, seed=0)
    for model, V in ((m0, None), (m1, data.V[tr])):
        fitted = model.predict(data.X[tr], V)
        assert np.all(fitted > 0)
        assert np.isclose(np.mean(data.y[tr] / fitted), 1.0, rtol=0.15)   # QLIKE first-order condition
    clone = pickle.loads(pickle.dumps(m1))
    np.testing.assert_array_equal(clone.predict(data.X[te], data.V[te]), m1.predict(data.X[te], data.V[te]))
    assert all(isinstance(v, np.ndarray) for v in m1.params_.values())


def test_neural_tuning_returns_epochs(synth, cfg):
    data = _with_lags(synth)
    local = {**cfg, "neural": NN, "families": {**cfg["families"],
             "tcn": {"n_candidates": 2, "space": {"hidden": [8], "lr": [3e-3], "weight_decay": [1e-4, 1e-2]}}}}
    fold = make_folds(data, local, local["walk_forward"])[0]
    it, va = fold.inner_train_idx, fold.val_idx
    rows = validation_table("tcn", data.X[it], data.V[it], data.y[it], data.X[va], data.V[va], data.y[va], local, 0)
    assert len(rows) == 2 and all(r["hp"]["epochs"] % NN["eval_every"] == 0 for r in rows)
    assert rows[0]["val_qlike"] <= rows[1]["val_qlike"]
    assert isinstance(make_model("tcn", rows[0]["hp"], local), NeuralModel)


def test_sequence_model_reads_only_the_return_path(synth, cfg):
    """Its forecast must not move when the summaries move, which is why P2 is skipped for it."""
    data = _with_lags(synth)
    local = {**cfg, "neural": NN}
    fold = make_folds(data, local, local["walk_forward"])[0]
    tr, te = fold.train_idx, fold.test_idx
    m0, _ = fit_pair("tcn", data.X[tr], data.V[tr], data.y[tr], {**HP["tcn"], "epochs": 50},
                     {**HP["tcn"], "epochs": 50}, local, seed=0)
    X = data.X[te].copy()
    X[:, HIST_COLS.index("rv_m")] *= 3.0
    np.testing.assert_array_equal(m0.predict(X), m0.predict(data.X[te]))


def test_gpu_round_robin_is_balanced(tmp_path, monkeypatch):
    from ivbehav.neural import _next_gpu
    monkeypatch.setenv("TMPDIR", str(tmp_path))
    assert [_next_gpu(2) for _ in range(6)] == [0, 1, 0, 1, 0, 1]
