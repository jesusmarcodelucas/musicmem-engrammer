"""
probe_basins.py -- visualize autoregressive "basins of attraction".

Teacher-forced recall only ever uses TRUE contexts. Autoregression feeds the
model its own outputs, so one wrong note produces a context that was never in
training (off-manifold). Whether the rollout recovers depends on whether that
off-manifold code still falls into the correct stored basin.

For each condition we train the model, then run an AR rollout per song and, at
each step, record:
  overlap_intended  Jaccard(active code from the model's own context,
                             active code from the TRUE context at that position)
                    -- how close we still are to the intended attractor.
  overlap_nearest   max Jaccard with ANY stored training code -- the nearest basin.
  nearest_ok        whether that nearest basin is the intended one (else we have
                    drifted into a DIFFERENT song's basin).
  acc               fraction of the 4 predicted pitches correct this step.

Outputs a figure (overlap & accuracy vs rollout depth) and a printed summary.

Run:
    python probe_basins.py                 # all three conditions, 120 epochs
    python probe_basins.py --epochs 60     # quicker
Needs: numpy, matplotlib (pip install matplotlib).
"""

import argparse
import numpy as np

import biomodel2 as B
from corpus import load_corpus, TRAIN_PATH, BAR_SIZE
K_BAR = 3


# ---------- data ----------
def real_songs():
    return [b for _, b in load_corpus(TRAIN_PATH)]


def random_songs(seed=0):
    barcounts = [len(b) for b in real_songs()]
    rng = np.random.default_rng(seed)

    def random_bar():
        slots = []
        while len(slots) < BAR_SIZE:
            dur = int(rng.choice([1, 2, 3, 4], p=[0.6, 0.3, 0.05, 0.05]))
            dur = min(dur, BAR_SIZE - len(slots))
            p = int(rng.integers(0, 12))
            slots.append((p, 1)); slots += [(p, 0)] * (dur - 1)
        return slots[:BAR_SIZE]

    return [[random_bar() for _ in range(nb)] for nb in barcounts]


# ---------- helpers ----------
def active(model, ctx):
    _, h = B.encode(model, ctx)
    return (h > 0).astype(float)


def stored_codes(model, songs):
    """All TRUE-context active codes, with (song_id, t) labels."""
    codes, owner = [], []
    for sid, bars in enumerate(songs):
        for t in range(K_BAR, len(bars)):
            codes.append(active(model, bars[t - K_BAR:t]))
            owner.append((sid, t))
    return np.array(codes), owner


def jaccard_to_all(a, S):
    inter = S @ a
    union = S.sum(1) + a.sum() - inter
    return inter / np.maximum(1.0, union)


# ---------- the probe ----------
def probe(model, songs, max_depth=12):
    S, owner = stored_codes(model, songs)
    owner_idx = {ow: i for i, ow in enumerate(owner)}
    rows = []                                  # (depth, ov_intended, ov_nearest, nearest_ok, acc)
    for sid, bars in enumerate(songs):
        ctx = [list(b) for b in bars[:K_BAR]]
        for t in range(K_BAR, len(bars)):
            a_gen = active(model, ctx[-K_BAR:])           # model's own (corrupted) code
            a_true = S[owner_idx[(sid, t)]]               # intended stored code
            ov_int = (a_gen @ a_true) / max(1.0, a_gen.sum() + a_true.sum() - a_gen @ a_true)
            j = jaccard_to_all(a_gen, S)
            near = int(np.argmax(j))
            ov_near = j[near]
            nearest_ok = int(owner[near] == (sid, t))
            _, h = B.encode(model, ctx[-K_BAR:])          # actual graded code
            preds = B.predict_bar(model, h)
            acc = np.mean([int(np.argmax(preds[i][0]) == bars[t][i][0]) for i in range(BAR_SIZE)])
            rows.append((t - K_BAR, ov_int, ov_near, nearest_ok, acc))
            ctx.append([(int(np.argmax(preds[i][0])), preds[i][1]) for i in range(BAR_SIZE)])
    return np.array(rows)


def _code(model, ctx):
    _, h = B.encode(model, ctx[-K_BAR:])
    return h


def by_depth(rows, col, max_depth=12):
    xs, ys = [], []
    for d in range(max_depth):
        m = rows[:, 0] == d
        if m.sum() >= 3:
            xs.append(d); ys.append(rows[m, col].mean())
    return np.array(xs), np.array(ys)


def train(songs, L, k, epochs, seed=0):
    B.LATENT = L; B.KWTA_FRAC = k / L
    B.set_ablation(B.AblationConfig())
    np.random.seed(seed)
    model, _ = B.train_model(songs, epochs=epochs, verbose=False)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--out", default="results/basins.png")
    args = ap.parse_args()

    rs, rnd = real_songs(), random_songs()
    conditions = [
        ("real  L=512", rs,  512, 30, "tab:green"),
        ("random L=512", rnd, 512, 30, "tab:orange"),
        ("random L=128", rnd, 128, 30, "tab:red"),
    ]

    results = {}
    print(f"{'condition':<14}{'ov(correct)':>12}{'ov(wrong)':>11}{'%nearest_ok':>13}")
    print("-" * 50)
    for name, songs, L, k, _ in conditions:
        m = train(songs, L, k, args.epochs)
        rows = probe(m, songs)
        results[name] = rows
        corr = rows[:, 1][rows[:, 4] > 0.99]          # overlap when step fully correct
        wrong = rows[:, 1][rows[:, 4] < 0.99]
        print(f"{name:<14}{(corr.mean() if len(corr) else float('nan')):>12.3f}"
              f"{(wrong.mean() if len(wrong) else float('nan')):>11.3f}"
              f"{100*rows[:,3].mean():>12.1f}%")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        for name, songs, L, k, c in conditions:
            rows = results[name]
            x, y = by_depth(rows, 1)
            ax1.plot(x, y, "-o", ms=3, color=c, label=name)
            x2, y2 = by_depth(rows, 4)
            ax2.plot(x2, y2, "-o", ms=3, color=c, label=name)
        ax1.set_title("code overlap with intended attractor")
        ax1.set_xlabel("autoregressive step (rollout depth)")
        ax1.set_ylabel("Jaccard(own code, true-context code)")
        ax1.set_ylim(0, 1); ax1.legend(fontsize=8); ax1.grid(alpha=.3)
        ax2.set_title("pitch accuracy")
        ax2.set_xlabel("autoregressive step (rollout depth)")
        ax2.set_ylabel("fraction correct"); ax2.set_ylim(0, 1)
        ax2.legend(fontsize=8); ax2.grid(alpha=.3)
        fig.tight_layout()
        from pathlib import Path
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=130)
        print(f"\nsaved figure -> {args.out}")
    except ImportError:
        print("\n(matplotlib not installed; skipped figure. pip install matplotlib)")


if __name__ == "__main__":
    main()
