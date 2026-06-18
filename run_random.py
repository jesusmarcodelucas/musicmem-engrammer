"""
run_random.py -- structure-vs-random control (Table C in the paper).

Trains the Dale-law memorizer on (a) the real corpus and (b) a size/onset-matched
random corpus (collision-free by construction), at matched (L, k). Reports TF and
AR. The expected result: TF saturates for both (now that the real corpus is also
collision-free), while AR shows the crossover -- real >= random at low capacity
(structural recovery), random -> 1.000 at high capacity (never leaves manifold).

    python run_random.py                      # L in {128,512,1024}, 120 ep
    python run_random.py --epochs 80

Writes results/random.json. Needs numpy, corpus.py, biomodel2.py, probe_basins.py.
"""
import argparse, json, time
import numpy as np
import biomodel2 as B
from corpus import RESULTS_DIR
from probe_basins import real_songs, random_songs


def collisions(songs):
    seen = {}; c = 0
    for bars in songs:
        for t in range(3, len(bars)):
            ctx = tuple(tuple(s) for bar in bars[t-3:t] for s in bar)
            tgt = tuple(tuple(s) for s in bars[t])
            if ctx in seen and seen[ctx] != tgt:
                c += 1
            seen[ctx] = tgt
    return c


def train_eval(songs, L, k, epochs, seed):
    B.LATENT = L; B.KWTA_FRAC = k / L
    B.set_ablation(B.AblationConfig())
    np.random.seed(seed)
    model, _ = B.train_model(songs, epochs=epochs, verbose=False)
    tf, ar = B.evaluate(model, songs)
    return float(tf.mean()), float(ar.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Lk", nargs="+", default=["128:15", "512:60", "1024:60"],
                    help="list of L:k pairs")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    real = real_songs(); rnd = random_songs(seed=args.seed)
    print(f"real collisions: {collisions(real)}   random collisions: {collisions(rnd)}")
    rows = []
    for pair in args.Lk:
        L, k = map(int, pair.split(":"))
        for name, songs in [("real", real), ("random", rnd)]:
            t0 = time.time()
            tf, ar = train_eval(songs, L, k, args.epochs, args.seed)
            r = dict(cond=name, L=L, k=k, TF=round(tf, 3), AR=round(ar, 3),
                     sec=round(time.time() - t0, 1))
            rows.append(r)
            print(f"{name:7s} L={L:5d} k={k:3d}  TF={tf:.3f}  AR={ar:.3f}  ({r['sec']}s)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(RESULTS_DIR / "random.json", "w"), indent=2)
    print(f"\nsaved {RESULTS_DIR/'random.json'}")


if __name__ == "__main__":
    main()
