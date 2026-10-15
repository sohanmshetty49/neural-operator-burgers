# neural-operator-burgers

Learning the solution operator of the 1D viscous Burgers equation from simulation data, and comparing three neural surrogates under identical data, optimiser and budget:

- **MLP** baseline (fixed grid)
- **Transformer** (grid points as tokens, Fourier coordinate features)
- **Fourier Neural Operator (FNO)** (spectral convolutions)

The task: map the initial condition `u(x, 0)` to `u(x, T=1)` for `u_t + u u_x = nu u_xx`, `nu = 0.01`, periodic domain `[0, 1)`.

## Results

Setup: 128-point grid, 1,000 training / 200 test trajectories, 100 epochs, Adam + OneCycle, relative-L2 loss, 3 seeds, CPU only. All numbers below come from `results/results.json`.

| Model | Params | Test relative L2 (mean ± std, 3 seeds) | Zero-shot on 256-pt grid | Inference, ms/sample (batch of 200, CPU) |
|---|---|---|---|---|
| Predict `u0` (identity) | – | 3.343 | – | – |
| MLP | 657,024 | 0.0380 ± 0.0003 | not possible (fixed grid) | 0.008 |
| Transformer | 135,233 | 0.0496 ± 0.0038 | 0.0496 | 1.44 |
| **FNO** | 139,745 | **0.0062 ± 0.0002** | 0.0062 | 0.087 |

- At about the same parameter count (~140K), the FNO's error is **8x lower than the Transformer's** and about **6x lower than a MLP with 4.7x more parameters**.
- Both resolution-agnostic models (FNO, Transformer) were trained on 128 points and evaluated on 256 points with no retraining; their error was unchanged to the reported precision. The MLP cannot take a different grid at all.
- Solver cost, same CPU: **8.8 ms/sample batched, 73.6 ms/sample for a single solve**. The FNO takes 0.087 ms/sample batched and 1.12 ms for a single call, i.e. roughly 100x and 65x faster respectively.

Figures: `results/predictions.png`, `results/training_curves.png`. Table: `results/summary.md`.

## Caveats (read these)

- **Small and easy problem.** Initial conditions are band-limited (32 Fourier modes), nu = 0.01 keeps shocks smooth, and the dataset is tiny. This shows the relative strengths of the architectures on this setup, not a general benchmark.
- **The zero-shot result is expected here**, because the fields are band-limited, so a 256-point grid contains no new content beyond the training band. It does not show generalisation to unseen frequencies or to different `nu`.
- **The Transformer was not tuned.** One configuration (d_model 64, 4 heads, 4 layers) and one learning rate were used; a larger or better-tuned Transformer may close part of the gap. The comparison is at equal budget, not at each model's best.
- **CPU timings** (a shared cloud container) are indicative; the speed-up ratios will differ on a GPU. The solver is a plain NumPy implementation.
- Trained on CPU. Nothing here uses CUDA.

## Solver

`solver.py`: pseudo-spectral, integrating-factor RK4 with 2/3 dealiasing. Checks in `tests/test_solver.py`:

- mean is conserved to ~1e-12
- matches the analytic heat-equation decay for tiny amplitude
- converges in space (128 vs 512 points) and time (dt vs dt/2) to < 1e-6 relative error
- initial conditions are identical across grid resolutions

## Run it

```bash
pip install -r requirements.txt
python -m pytest -q tests
python run_experiment.py --out results      # about 30 min on CPU (the Transformer dominates)
python plot_results.py results
```

Quick smoke run: `python run_experiment.py --ntrain 64 --ntest 32 --epochs 3 --seeds 0 --out /tmp/smoke`.

## Layout

```
solver.py          Burgers solver + data generation
models.py          MLP, TransformerOperator, FNO1d
run_experiment.py  training, evaluation, timing
plot_results.py    figures and summary table
tests/             solver and model tests
results/           results.json, predictions.npz, figures
```
