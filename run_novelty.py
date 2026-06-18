"""
run_novelty.py -- two experiments with novelty-gated plasticity.

Plasticity is gated by per-window surprise: each weight update is scaled by a
sigmoid of that window's prediction error (RPE), so already-known windows stop
being rewritten and only novel/surprising ones drive learning. Knobs live in
biomodel2: NOVELTY_GATE (on/off), NOVELTY_THR (half-open surprise level),
NOVELTY_TEMP (gate softness).

Experiment 1 -- gate during normal training (does it cure the late AR dip?)
    Trains the full corpus with the gate on, printing AR at checkpoints. The
    prediction: the gate freezes near-correct windows so late updates stop
    perturbing the shared code, and AR should plateau at its peak instead of
    dipping after ~epoch 40.
        python run_novelty.py exp1 --epochs 120 --ar-every 10
        python run_novelty.py exp1 --epochs 120 --ar-every 10 --no-gate   # control

Experiment 2 -- incremental new song (does novelty-gating avoid forgetting?)
    Trains a base set of (N-1) songs to convergence, then introduces ONE new
    (held-out) song with the gate ON, and tracks how well the OLD songs survive
    vs how fast the NEW one is learned. The gate should open for the novel song
    (high surprise) and stay ~closed for the consolidated base (low surprise),
    so consolidation protects the old memories.
        python run_novelty.py exp2 --latent 512 --base-epochs 120 --new-epochs 40
        python run_novelty.py exp2 --no-gate     # control: ungated -> expect forgetting

Needs: numpy.
"""

import argparse
import numpy as np

import biomodel2 as B
from corpus import load_corpus, TRAIN_PATH, NOVEL_PATH


def tf_ar(model, bars_list):
    tf, ar = B.evaluate(model, bars_list)
    return float(tf.mean()), float(ar.mean())


# ----------------------------------------------------------------------
def exp1(args):
    """Novelty-gated plasticity during normal full-corpus training."""
    B.LATENT = args.latent
    B.KWTA_FRAC = args.k / args.latent
    B.NOVELTY_GATE = not args.no_gate
    B.NOVELTY_THR = args.thr
    B.NOVELTY_TEMP = args.temp
    B.set_ablation(B.AblationConfig())
    np.random.seed(args.seed)

    train = [b for _, b in load_corpus(TRAIN_PATH)]
    print(f"EXP1  gate={'ON' if B.NOVELTY_GATE else 'OFF'}  "
          f"L={args.latent} k={args.k} thr={args.thr} temp={args.temp}")
    model, hist = B.train_model(train, epochs=args.epochs, verbose=True,
                                log_every=args.epochs, ar_every=args.ar_every)
    print("\nAR trajectory (epoch, TF, AR):")
    for ep, tf, ar in hist["ar_checkpoints"]:
        print(f"  ep {ep:03d}:  TF={tf:.4f}  AR={ar:.4f}")
    print("\nWatch whether AR peaks early then dips (ungated) or plateaus (gated).")


# ----------------------------------------------------------------------
def exp2(args):
    """Incremental learning: memorize base set, then add ONE new song."""
    B.LATENT = args.latent
    B.KWTA_FRAC = args.k / args.latent
    B.set_ablation(B.AblationConfig())
    np.random.seed(args.seed)

    train = load_corpus(TRAIN_PATH)
    novel = load_corpus(NOVEL_PATH)
    base = [b for _, b in train]                       # the old songs (50)
    new_name, new_bars = novel[0]                      # the incoming song
    n_total = len(base) + 1

    # ---- phase A: learn the base set to convergence (gate OFF) ----
    B.NOVELTY_GATE = False
    # build a model sized for base+1 song so the recognition head has room
    model = B.init_model(n_total)
    print(f"EXP2  base={len(base)} songs -> +1 new ('{new_name}')  "
          f"gate={'ON' if not args.no_gate else 'OFF'}  L={args.latent} k={args.k}")
    print(f"\n[phase A] consolidating base set ({args.base_epochs} epochs)...")
    for ep in range(args.base_epochs):
        for sid in np.random.permutation(len(base)):
            B.train_on_song(model, base[sid], sid)
    base_tf0, base_ar0 = tf_ar(model, base)
    print(f"  base after phase A:  TF={base_tf0:.4f}  AR={base_ar0:.4f}")

    # ---- phase B: introduce the new song (gate ON unless --no-gate) ----
    B.NOVELTY_GATE = not args.no_gate
    B.NOVELTY_THR = args.thr
    B.NOVELTY_TEMP = args.temp
    new_sid = len(base)                                # id of the new song
    print(f"\n[phase B] introducing new song with gate "
          f"{'ON' if B.NOVELTY_GATE else 'OFF'} ({args.new_epochs} epochs)...")
    print(f"  {'epoch':>6}{'base_TF':>10}{'base_AR':>10}{'new_TF':>9}{'new_AR':>9}")
    for ep in range(args.new_epochs):
        # interleave the new song with occasional base replay? -> pure new-song
        # exposure is the harder, more revealing test, so train only the new song:
        B.train_on_song(model, new_bars, new_sid)
        if ep % max(1, args.new_epochs // 10) == 0 or ep == args.new_epochs - 1:
            btf, bar = tf_ar(model, base)
            ntf, nar = tf_ar(model, [new_bars])
            print(f"  {ep:>6}{btf:>10.4f}{bar:>10.4f}{ntf:>9.4f}{nar:>9.4f}")

    btf, bar = tf_ar(model, base)
    ntf, nar = tf_ar(model, [new_bars])
    print(f"\nRESULT:")
    print(f"  base TF: {base_tf0:.4f} -> {btf:.4f}   (drop = {base_tf0-btf:+.4f}; "
          f"forgetting if large negative)")
    print(f"  base AR: {base_ar0:.4f} -> {bar:.4f}")
    print(f"  new song learned: TF={ntf:.4f}  AR={nar:.4f}")
    print("\nGate ON should protect base (small drop) while still learning the new "
          "song; gate OFF (control) should show larger base forgetting.")


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="novelty-gated plasticity experiments")
    sub = ap.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--latent", type=int, default=512)
    common.add_argument("--k", type=int, default=30)
    common.add_argument("--seed", type=int, default=0)
    common.add_argument("--thr", type=float, default=0.05, help="gate half-open surprise")
    common.add_argument("--temp", type=float, default=0.02, help="gate softness")
    common.add_argument("--no-gate", action="store_true", help="disable the gate (control)")

    p1 = sub.add_parser("exp1", parents=[common], help="gate during training (AR dip)")
    p1.add_argument("--epochs", type=int, default=120)
    p1.add_argument("--ar-every", type=int, default=10)

    p2 = sub.add_parser("exp2", parents=[common], help="incremental new song (forgetting)")
    p2.add_argument("--base-epochs", type=int, default=120)
    p2.add_argument("--new-epochs", type=int, default=40)

    args = ap.parse_args()
    (exp1 if args.cmd == "exp1" else exp2)(args)


if __name__ == "__main__":
    main()
