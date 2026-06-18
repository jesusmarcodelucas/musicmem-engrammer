"""
run_sweep.py -- capacity sweep for the Dale-law memorizer (Table A in the paper).

Sweeps latent size L at FIXED sparsity (k/L constant). TF/recognition/novelty
should saturate; only AR (autoregressive recall) scales with L.

    python run_sweep.py                       # L in {64,128,256,512,1024}, 120 ep
    python run_sweep.py --epochs 80 --seeds 0 1 2
    python run_sweep.py --kfrac 0.059         # sparsity (default 0.059, headline)

Writes results/sweep.json. Needs: numpy, corpus.py, biomodel2.py, data/*.json.
"""
import argparse, json, time
from pathlib import Path
import numpy as np
import biomodel2 as B
from corpus import RESULTS_DIR


def one_run(L, kfrac, epochs, seed):
    B.KWTA_FRAC = kfrac
    B.set_ablation(B.AblationConfig())
    _, _, out = B.run(epochs=epochs, latent=L, seed=seed, verbose=False, novelty=True)
    nv = out["novelty"]
    return {
        "L": L, "k": int(round(kfrac * L)), "kfrac": kfrac, "seed": seed,
        "TF": out["tf_pitch_acc"], "AR": out["ar_pitch_acc"],
        "recog": out["recognition_acc"],
        "novelty": f"{nv['novel_detected']}/{nv['n_novel']}",
        "nov_thr": nv["threshold"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ls", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    ap.add_argument("--kfrac", type=float, default=0.117)  # paper: k/L~0.117 (L=1024 caps k=60)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = []
    for L in args.Ls:
        for s in args.seeds:
            t0 = time.time()
            r = one_run(L, args.kfrac, args.epochs, s)
            r["sec"] = round(time.time() - t0, 1)
            rows.append(r)
            print(f"L={L:5d} k={r['k']:3d} seed={s}  TF={r['TF']:.3f}  "
                  f"AR={r['AR']:.3f}  recog={r['recog']:.3f}  "
                  f"nov={r['novelty']}  ({r['sec']}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = Path(args.out) if args.out else RESULTS_DIR / "sweep.json"
    json.dump(rows, open(out, "w"), indent=2)
    print(f"\nsaved {out}")

    # mean +/- sd over seeds, per L
    print("\n== AR mean+/-sd over seeds ==")
    for L in args.Ls:
        ar = [r["AR"] for r in rows if r["L"] == L]
        tf = [r["TF"] for r in rows if r["L"] == L]
        print(f"  L={L:5d}: TF={np.mean(tf):.3f}  AR={np.mean(ar):.3f}"
              + (f" +/- {np.std(ar):.3f}" if len(ar) > 1 else ""))


if __name__ == "__main__":
    main()
