#!/usr/bin/env python3
"""Encoding comparison: attack (13/slot) vs dual (24/slot) vs factored (14/slot),
on the real corpus and a size/onset-matched random corpus, across capacities.

Reproduces results/enc_grid.json and the summary table in the preprint
(autoregressive recall as a function of input encoding and latent size).

Usage:
    python run_encoding_compare.py                 # default grid, 5 seeds
    python run_encoding_compare.py --seeds 3       # fewer seeds (faster)
"""
import argparse, json
import numpy as np
import biomodel2 as B
from corpus import RESULTS_DIR
from probe_basins import real_songs, random_songs

KMAP = {64: 8, 128: 15, 256: 30, 512: 60, 1024: 60}   # k/L ~ 0.117 (0.059 at L=1024)


def trial(mode, corpus, L, seed, epochs=120):
    k = KMAP[L]
    B.LATENT = L
    B.KWTA_FRAC = k / L
    B.set_ablation(B.AblationConfig(input_mode=mode))
    np.random.seed(seed)
    songs = real_songs() if corpus == "real" else random_songs(seed=seed)
    m, _ = B.train_model(songs, epochs=epochs, verbose=False)
    tf, ar = B.evaluate(m, songs)
    return round(float(tf.mean()), 3), round(float(ar.mean()), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=["attack", "dual", "factored"])
    ap.add_argument("--Ls", nargs="+", type=int, default=[128, 256, 512])
    ap.add_argument("--out", default=str(RESULTS_DIR / "enc_grid.json"))
    args = ap.parse_args()

    try:
        grid = json.load(open(args.out))
    except FileNotFoundError:
        grid = {}

    for L in args.Ls:
        for mode in args.modes:
            for corpus in ["real", "random"]:
                for s in range(args.seeds):
                    key = f"{mode}_{corpus}_L{L}_s{s}"
                    if key in grid:
                        continue
                    tf, ar = trial(mode, corpus, L, s)
                    grid[key] = {"TF": tf, "AR": ar}
                    json.dump(grid, open(args.out, "w"), indent=2)
                    print(f"{mode:9s} {corpus:6s} L{L:<4d} s{s}: TF={tf:.3f} AR={ar:.3f}")

    # summary
    def cell(mode, corpus, L):
        ar = [grid[f"{mode}_{corpus}_L{L}_s{s}"]["AR"] for s in range(args.seeds)
              if f"{mode}_{corpus}_L{L}_s{s}" in grid]
        tf = [grid[f"{mode}_{corpus}_L{L}_s{s}"]["TF"] for s in range(args.seeds)
              if f"{mode}_{corpus}_L{L}_s{s}" in grid]
        return np.mean(tf), np.mean(ar), np.std(ar)

    print("\nAR mean +/- sd (TF mean) over seeds")
    for L in args.Ls:
        for mode in args.modes:
            tr, arr, sdr = cell(mode, "real", L)
            tn, arn, sdn = cell(mode, "random", L)
            print(f"  L{L:<4d} {mode:9s} real {arr:.3f}+/-{sdr:.3f} (TF{tr:.2f}) | "
                  f"random {arn:.3f}+/-{sdn:.3f} (TF{tn:.2f})")


if __name__ == "__main__":
    main()
