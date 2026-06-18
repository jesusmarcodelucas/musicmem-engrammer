"""
run_ablation_table.py -- leave-one-out ablation at fixed (L, k) (Table B).

Each mechanism is turned off in turn (plus the additive +encoder_learn probe).
Saves results/ablation.json. Uses a matched epoch budget across all configs so
the deltas are comparable; the full model reaches TF=1.000 only by ~120 epochs
(see run_sweep.py), so report the budget you used.

    python run_ablation_table.py                       # L=256 k=30 80ep
    python run_ablation_table.py --epochs 120 --L 256 --k 30

Needs numpy, corpus.py, biomodel2.py, data/*.json.
"""
import argparse, json, time
import biomodel2 as B
from corpus import RESULTS_DIR

CONFIGS = [
    ("full", {}),
    ("dale=False", dict(dale=False)),
    ("multisyn=False", dict(multisyn=False)),
    ("consolidation=False", dict(consolidation=False)),
    ("ltd=False", dict(ltd=False)),
    ("kwta=False", dict(kwta=False)),
    ("sparse_wiring=False", dict(sparse_wiring=False)),
    ("div_inh=False", dict(div_inh=False)),
    ("classifier=False", dict(classifier=False)),
    ("+encoder_learn", dict(encoder_learn=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--k", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    B.LATENT = args.L; B.KWTA_FRAC = args.k / args.L

    rows = []
    for label, kw in CONFIGS:
        B.set_ablation(B.AblationConfig(**kw))
        t0 = time.time()
        _, _, o = B.run(epochs=args.epochs, latent=args.L, seed=args.seed,
                        verbose=False, novelty=True)
        nv = o["novelty"]
        r = dict(config=label, TF=round(o["tf_pitch_acc"], 3),
                 AR=round(o["ar_pitch_acc"], 3),
                 recog=(None if o["recognition_acc"] is None
                        else round(o["recognition_acc"], 3)),
                 novelty=f"{nv['novel_detected']}/{nv['n_novel']}",
                 sec=round(time.time() - t0, 1))
        rows.append(r)
        print(f"{label:24s} TF={r['TF']:.3f} AR={r['AR']:.3f} "
              f"recog={r['recog']} nov={r['novelty']} ({r['sec']}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RESULTS_DIR / "ablation.json", "w"), indent=2)
    print(f"\nsaved {RESULTS_DIR/'ablation.json'}  (L={args.L} k={args.k} "
          f"{args.epochs}ep seed={args.seed})")


if __name__ == "__main__":
    main()
