"""Make figures from results/results.json and results/predictions.npz."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main(out="results"):
    out = Path(out)
    res = json.loads((out / "results.json").read_text())
    pred = np.load(out / "predictions.npz")
    names = list(res["models"])
    n = pred["u0"].shape[1]
    x = np.arange(n) / n

    # 1) Predictions vs. truth for the first test sample
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    ax[0].plot(x, pred["u0"][0], label="u(x, 0)")
    ax[0].plot(x, pred["truth"][0], "k", lw=2, label="solver u(x, T)")
    for nm in names:
        ax[0].plot(x, pred[nm][0], "--", label=nm)
    ax[0].set_xlabel("x"); ax[0].set_title("Test sample 0"); ax[0].legend(fontsize=7)
    for nm in names:
        err = np.abs(pred[nm][0] - pred["truth"][0])
        ax[1].semilogy(x, err + 1e-12, label=nm)
    ax[1].set_xlabel("x"); ax[1].set_title("Absolute error"); ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "predictions.png", dpi=150); plt.close(fig)

    # 2) Test error during training (seed 0)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for nm in names:
        ep, e = zip(*res["models"][nm]["curve_seed0"])
        ax.semilogy(ep, e, label=nm)
    ax.axhline(res["baseline_identity_rel_l2"], color="gray", ls=":", label="predict u0")
    ax.set_xlabel("epoch"); ax.set_ylabel("test relative L2"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=150); plt.close(fig)

    # 3) Summary table
    lines = ["| Model | Params | Test rel-L2 (mean ± std, 3 seeds) | Zero-shot 2x grid | Inference ms/sample (batch) |",
             "|---|---|---|---|---|"]
    for nm in names:
        m = res["models"][nm]
        zs = "n/a (fixed grid)" if m["zero_shot_2x_grid_rel_l2_mean"] is None else f'{m["zero_shot_2x_grid_rel_l2_mean"]:.4f}'
        lines.append(f'| {nm} | {m["params"]:,} | {m["rel_l2_mean"]:.4f} ± {m["rel_l2_std"]:.4f} | {zs} | {m["infer_ms_per_sample_batch200"]:.3f} |')
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
