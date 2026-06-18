"""
run_L1024.py -- run the Dale-law memorizer at a chosen latent size locally.

This is a thin convenience wrapper around biomodel2.py (the Dale-law, multi-
synapse scalar model). It sets the latent size and the k-WTA active count
(k/L sparsity), optionally applies the init=0.5 decoder lever, trains, and
prints TF / AR pitch accuracy, recognition, and novelty.

Files expected next to this script:
    corpus.py
    biomodel2.py
    data/songs_train.json
    data/songs_novel.json

Requirements:  python 3.9+, numpy.

Examples
--------
    python run_L1024.py                       # L=1024, k=60, 120 epochs
    python run_L1024.py --k 30                 # sparser code (k/L = 0.029)
    python run_L1024.py --init 0.5             # apply the decoder init lever
    python run_L1024.py --latent 2048 --k 80   # go bigger
    python run_L1024.py --epochs 200 --seed 1

Notes
-----
* Keep k/L below ~0.12 (the memorization/sparsity threshold). At L=1024 that
  means k <= ~120; the default k=60 (k/L=0.059) is comfortably sparse.
* L=1024 is ~4x the per-step cost of L=256; a 120-epoch run takes a few minutes
  on a laptop CPU. Use --epochs 60 for a quicker look.
* To run the stochastic multi-contact model (v3) instead, see the bottom of this
  file or run:  python biomodel3.py --latent 1024 --epochs 80
"""

import argparse
import numpy as np

import biomodel2 as B


def main():
    ap = argparse.ArgumentParser(description="Dale-law memorizer at chosen latent size")
    ap.add_argument("--latent", type=int, default=1024, help="latent units L (default 1024)")
    ap.add_argument("--k", type=int, default=60,
                    help="k-WTA active units; sparsity is k/L (default 60)")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", type=float, default=None,
                    help="decoder init value, e.g. 0.5 (default: small random, the "
                         "biomodel2 default). 0.5 gives LTD more room early.")
    ap.add_argument("--ar-every", type=int, default=0,
                    help="print autoregressive recall every N epochs (0 = only at end)")
    args = ap.parse_args()

    # set capacity and sparsity (read live by the model)
    B.LATENT = args.latent
    B.KWTA_FRAC = args.k / args.latent

    # optional decoder-init lever (the +0.09 AR fix we found): center the Dale
    # decoder weights at `init` instead of ~0, so depression has room from the start.
    if args.init is not None:
        _orig_init = B.init_model

        def _init(n_songs):
            m = _orig_init(n_songs)
            N = B.LATENT
            for i in range(B.BAR_SIZE):
                m[f"W_out{i}"]["W"] = np.clip(
                    args.init * (1.0 + 0.1 * np.random.randn(B.OUTPUT_SLOT, N)),
                    0.0, B.W_MAX)
            return m

        B.init_model = _init

    print(f"Dale model | L={args.latent}  k={args.k} (k/L={args.k/args.latent:.3f})  "
          f"epochs={args.epochs}  seed={args.seed}  "
          f"init={'default' if args.init is None else args.init}")

    B.set_ablation(B.AblationConfig())                 # full Dale model
    model, hist, summary = B.run(epochs=args.epochs, latent=args.latent,
                                 seed=args.seed, verbose=True, novelty=True,
                                 ar_every=args.ar_every)

    print("\n================ RESULT ================")
    print(f"  TF pitch accuracy : {summary['tf_pitch_acc']:.3f}")
    print(f"  AR pitch accuracy : {summary['ar_pitch_acc']:.3f}")
    print(f"  recognition       : {summary['recognition_acc']:.3f}")
    nv = summary.get("novelty", {})
    if nv:
        print(f"  novelty detected  : {nv['novel_detected']}/{nv['n_novel']} "
              f"(threshold {nv['threshold']:.3f})")
    print("========================================")


if __name__ == "__main__":
    main()
