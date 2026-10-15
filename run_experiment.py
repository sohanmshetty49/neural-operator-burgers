"""Train and compare MLP / Transformer / FNO surrogates for 1D viscous Burgers.

    python run_experiment.py                 # full run (~15-25 min on 2 CPU cores)
    python run_experiment.py --epochs 3 --ntrain 64 --ntest 32 --seeds 0   # smoke test

Protocol (fixed before looking at test results; no hyperparameter tuning on the test set):
  * data: u0 ~ Gaussian random field, u(T=1) from the spectral solver in solver.py
  * train/test: disjoint random initial conditions (different seeds), 128-point grid
  * loss/metric: relative L2 error  ||pred - true|| / ||true||  per sample, averaged
  * 3 independent training seeds per model; mean and std reported
  * extra test: zero-shot evaluation on a 256-point grid (never trained on)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import models
import solver

torch.set_num_threads(2)


def rel_l2(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return (pred - true).flatten(1).norm(dim=1) / true.flatten(1).norm(dim=1)


def train(model, xtr, ytr, xte, yte, epochs, seed, batch=32, lr=1e-3):
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * ((len(xtr) + batch - 1) // batch))
    curve = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g)
        for i in range(0, len(xtr), batch):
            idx = perm[i : i + batch]
            loss = rel_l2(model(xtr[idx]), ytr[idx]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        model.eval()
        with torch.no_grad():
            curve.append((ep + 1, float(rel_l2(model(xte), yte).mean())))
    return time.time() - t0, curve


@torch.no_grad()
def infer_ms_per_sample(model, x, reps=5):
    model.eval()
    model(x)
    t0 = time.perf_counter()
    for _ in range(reps):
        model(x)
    return (time.perf_counter() - t0) / reps / len(x) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--ntrain", type=int, default=1000)
    ap.add_argument("--ntest", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    u0_tr, uT_tr = solver.make_dataset(a.ntrain, a.n, seed=1000)
    t_solve_batch = time.perf_counter() - t0
    u0_te, uT_te = solver.make_dataset(a.ntest, a.n, seed=2000)
    # same test initial conditions on a 256-point grid, for the zero-shot resolution test
    c_te = solver.sample_coeffs(a.ntest, 2000)
    u0_hi = solver.initial_conditions(c_te, 2 * a.n)
    uT_hi = solver.solve(u0_hi)
    # single-sample solver latency (batch of 1), median of 5
    one = []
    for i in range(5):
        t = time.perf_counter()
        solver.solve(u0_te[i : i + 1])
        one.append(time.perf_counter() - t)
    solver_ms_batched = t_solve_batch / a.ntrain * 1e3
    solver_ms_single = float(np.median(one)) * 1e3

    T = lambda z: torch.tensor(z, dtype=torch.float32)
    xtr, ytr, xte, yte, xhi, yhi = map(T, (u0_tr, uT_tr, u0_te, uT_te, u0_hi, uT_hi))
    baseline = float(rel_l2(xte, yte).mean())  # "predict u(T) = u0"

    makers = {
        "MLP": lambda: models.MLP(a.n),
        "Transformer": lambda: models.TransformerOperator(),
        "FNO": lambda: models.FNO1d(),
    }
    results = {"config": vars(a), "solver": {"ms_per_sample_batched": solver_ms_batched, "ms_per_sample_single": solver_ms_single},
               "baseline_identity_rel_l2": baseline, "models": {}}
    preds = {}
    for name, make in makers.items():
        errs, hi_errs, times, curves = [], [], [], []
        for seed in a.seeds:
            torch.manual_seed(seed)
            m = make()
            secs, curve = train(m, xtr, ytr, xte, yte, a.epochs, seed)
            m.eval()
            with torch.no_grad():
                errs.append(float(rel_l2(m(xte), yte).mean()))
                if name != "MLP":
                    hi_errs.append(float(rel_l2(m(xhi), yhi).mean()))
                if seed == a.seeds[0]:
                    preds[name] = m(xte[:4]).numpy()
            times.append(secs)
            curves.append(curve)
            print(f"{name:12s} seed {seed}: test rel-L2 {errs[-1]:.4f}  train {secs:.0f}s", flush=True)
        results["models"][name] = {
            "params": models.count_params(m),
            "rel_l2_mean": float(np.mean(errs)), "rel_l2_std": float(np.std(errs)), "rel_l2_per_seed": errs,
            "zero_shot_2x_grid_rel_l2_mean": float(np.mean(hi_errs)) if hi_errs else None,
            "train_seconds_mean": float(np.mean(times)),
            "infer_ms_per_sample_batch200": infer_ms_per_sample(m, xte),
            "infer_ms_per_sample_single": infer_ms_per_sample(m, xte[:1], reps=20),
            "curve_seed0": curves[0],
        }
        (out / "results.json").write_text(json.dumps(results, indent=2))
    np.savez(out / "predictions.npz", u0=u0_te[:4], truth=uT_te[:4], **{k: v for k, v in preds.items()})
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: (v["rel_l2_mean"], v["rel_l2_std"]) for k, v in results["models"].items()}, indent=1))


if __name__ == "__main__":
    main()
