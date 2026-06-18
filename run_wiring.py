"""
run_wiring.py -- input-connectivity (wiring-density) sweep for the Dale model.

The encoder is a FIXED random projection; here we make its *wiring* sparse by
masking each latent unit to a random subset of inputs at a given density, and
sweep that density. k-WTA activity sparsity is unchanged. Reports TF/AR/recog/
novelty and the known-novel separation in sigma.

    python run_wiring.py                      # L=512 k=60, densities 1..0.05
    python run_wiring.py --L 256 --k 30

Writes results/wiring.json. Needs numpy, corpus.py, biomodel2.py, data/*.json.
"""
import argparse, json, time
import numpy as np
import biomodel2 as B
from corpus import RESULTS_DIR


def sigma_sep(nv):
    novel = np.array([r["rpe"] for r in nv["rows"]])
    return float((novel.mean() - nv["known_rpe_mean"]) / max(1e-9, nv["known_rpe_std"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dens", type=float, nargs="+",
                    default=[1.0, 0.25, 0.15, 0.10, 0.05])
    args = ap.parse_args()
    B.LATENT = args.L; B.KWTA_FRAC = args.k / args.L

    rows = []
    for d in args.dens:
        cfg = B.AblationConfig(conn_density=d, sparse_wiring=(d < 1.0))
        B.set_ablation(cfg)
        t0 = time.time()
        _, _, o = B.run(epochs=args.epochs, latent=args.L, seed=args.seed,
                        verbose=False, novelty=True)
        nv = o["novelty"]
        r = dict(density=d, TF=round(o["tf_pitch_acc"], 3),
                 AR=round(o["ar_pitch_acc"], 3),
                 recog=round(o["recognition_acc"], 3),
                 novelty=f"{nv['novel_detected']}/{nv['n_novel']}",
                 sigma=round(sigma_sep(nv), 1), sec=round(time.time() - t0, 1))
        rows.append(r)
        print(f"density={d:5.2f}  TF={r['TF']:.3f}  AR={r['AR']:.3f}  "
              f"recog={r['recog']:.3f}  nov={r['novelty']}  sep={r['sigma']}sigma  "
              f"({r['sec']}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RESULTS_DIR / "wiring.json", "w"), indent=2)
    print(f"\nsaved {RESULTS_DIR/'wiring.json'}  (L={args.L} k={args.k})")


if __name__ == "__main__":
    main()
