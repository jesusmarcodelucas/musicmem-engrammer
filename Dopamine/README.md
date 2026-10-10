# ENGRAMMER-DOPA: novelty from failed identification in sparse latent codes

Code, results and LaTeX source of the preprint

> Marco de Lucas, J. (2026). *ENGRAMMER-DOPA: Novelty from failed identification in sparse neural representations. A signed-XOR mechanism for dopamine-like memory allocation* (Version 0.8.2) [Preprint]. Zenodo.

In a signed-XOR autoencoder the XOR output (input vs. reconstruction) projects back to the latent neurons, where it instructs plasticity. With the inhibition set by the first response held fixed, the unexplained input bits also recruit latent units beyond the k-sparse code. Their number, ν, is a novelty signal. A dopamine-like pulse triggered by ν biases the competition for the final code towards little-used neurons (homeostatic excitability), which decides *where* a new item is stored. All learning is signed-XOR.

## Layout

```
Dopamine/
├── paper/     ENGRAMMER_DOPA_v0.8.2.tex / .pdf and the four figures (fig_v081_*.pdf)
├── code/      model, experiments, analysis and plotting scripts (Python 3, NumPy, Matplotlib)
└── results/   runs used in the paper
    ├── songs/   melodies, 30 seeds: pitch-only (main) and pitch+onset (replication), with reports
    ├── toy/     synthetic world, 20 seeds, full configuration grid + fixed-trigger ablations + v0.7 feedforward reference
    ├── mnist/   class-incremental MNIST, seeds 1–11 (seed 1 = selection): selected configuration (c⁻=0, λ=0.2, ν_th=1),
    │            error-count triggers (…_t2.pkl), ν distribution, and summaries of the whole grid (*_summary2.pkl)
    └── extra/   error count vs ν in the synthetic world, cost of one-off noise inputs, statistics output
```

The melody corpus is read from the root of this repository (`corpus.py`, `data/`). Set `MUSICMEM=/path/to/musicmem-engrammer` to use another copy. `code/mnist14.npz` is MNIST binarised at 0.5 and max-pooled to 14×14 (regenerate with `prepare_mnist.py`).

The full MNIST grid (c⁻ ∈ {0, 0.5} × λ ∈ {0.05, 0.2} × ν_th ∈ {1, 2, 4}, ≈120 MB of raw results) is not included; only the per-configuration summaries are. `run_all.sh mnist` regenerates it.

## Reproducing

All commands from `code/`:

```bash
bash quick_check.sh                       # ~2 min smoke test (melodies, MNIST, synthetic world)
NPROC=8 bash run_all.sh [songs|toy|mnist|all]   # all runs, ≈20 core-hours; writes code/results/ and code/logs/
python stats.py                           # statistics quoted in Sections 3–4 (bootstrap 95% CIs)
python plot_v081.py                       # Figures 2 and 3 (synthetic world, MNIST); FIGDIR=… to choose the folder
python plot_songs.py ../results/songs/songs_results_pitch.pkl fig_v081_songs.pdf              # Figure 4
python plot_songs_controls.py ../results/songs/songs_results_pitch.pkl fig_v081_songs_controls.pdf   # Figure 5
python analyse_songs.py ../results/songs/songs_results_pitch.pkl   # Tables 3–4 and Section 5 numbers
ENGDOPA_LAM=0.2 ENGDOPA_INH=kth ENGDOPA_CMINUS=0 python toy_errp2.py 1 2 3   # error count vs ν (synthetic world)
ENGDOPA_LAM=0.2 ENGDOPA_INH=kth ENGDOPA_CMINUS=0 python toy_noise.py 1 2 3   # pulses and units taken by one-off noise
```

`stats.py` and `plot_v081.py` read `../results/` by default; set `RESULTS=results` to use the output of `run_all.sh`.

## Main settings

| Variable | Meaning | Paper |
|---|---|---|
| `ENGDOPA_LAM` | gain λ of the XOR output on latent activity | 0.2 (0 = feedforward case of v0.7) |
| `ENGDOPA_INH` | inhibition: `kth` = level of the k-th initial winner; `max` = γ·max a | `kth` |
| `ENGDOPA_CMINUS` | depression of excess units (c⁻) | 0 |
| `NU_THR` | novelty threshold ν_th, in units of k | 8 (synthetic world), 1 (MNIST); melodies use ν > 0 |
| `E_THRS` | error-count triggers Σ C⁺ > ε (control) | 2,4,8 (synthetic world); 2,5,10 (MNIST) |

Selection rules: c⁻ and λ, and the MNIST threshold, were chosen on seed 1 (`select_config.py`); the synthetic-world threshold is the lowest value with ≤ 3 % false alarms on familiar presentations.

## Files in `code/`

- `engdopa.py` — encoder (provisional code, reconstruction, XOR output on latent activity, ν, allocation, signed-XOR learning) and transition predictor
- `world.py`, `toy.py`, `exp_toy.py`, `exp_toy_fix.py`, `analyse_toy*.py` — synthetic world (Section 3)
- `bench.py`, `exp_bench.py`, `exp_bench2.py`, `nu_mnist.py`, `analyse_bench*.py`, `prepare_mnist.py`, `select_config.py` — MNIST (Section 4)
- `songs_data.py`, `songs_net.py`, `exp_songs.py`, `analyse_songs.py`, `plot_songs*.py` — melodies (Section 5)
- `stats.py`, `plot_v081.py`, `toy_errp2.py`, `toy_noise.py` — statistics, figures and extra analyses of v0.8.1–0.8.2
