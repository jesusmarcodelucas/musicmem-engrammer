"""
biomodel2 -- Dale's-law, multi-synapse music memorizer (from scratch).

A clean re-implementation of the children's-song memorizer with a stricter
biological substrate than the original `biomodel.py`. Same task -- read 3 bars
of context (4/4, 4 beats each), predict the next bar's 4 beats -- but every
mechanism the brief asked for is enforced explicitly and is independently
ablatable.

What is different from biomodel.py
----------------------------------
1. DALE'S LAW. Every neuron is either excitatory or inhibitory, fixed at birth.
   All synaptic *weights are non-negative*; a presynaptic neuron's TYPE (not the
   sign of a weight) decides whether it adds to or subtracts from a postsynaptic
   sum. The old model used freely-signed weights. (flag: dale)

2. MULTI-SYNAPSE / QUANTAL WEIGHTS. A connection between two neurons is a bundle
   of up to N_SYN_MAX = 10 discrete contacts, each able to grow to strength 1.0,
   so an effective weight up to ~10 is built from several synapses rather than
   one large one. LTP fills the current contact toward 1.0 before recruiting the
   next; the integer part of the weight is the number of formed contacts.
   (flag: multisyn)

3. REPETITION-DRIVEN LTP + CONSOLIDATION. Each pre->post co-potentiation counts
   as a repetition; after CONSOLIDATE_REPS repetitions a newly formed contact is
   *consolidated* (locked). Consolidated contacts set a floor that depression
   (LTD) cannot fall below -- "once consolidated, not easy to drop." There is a
   hard ceiling W_MAX. (flags: consolidation, ltd)

4. INPUT WIRING SPARSITY ~10%. Each latent neuron keeps a fixed random ~10% of
   its possible input synapses (cortical local connection probability / fly
   mushroom-body fan-in). (flag: sparse_wiring, knob: conn_density)

5. k-WTA AT ~15%. The active code is the top ~15% of latent units
   (256 latent -> 38 active, in the requested 30-50 band). (flag: kwta)

6. SIMPLER INPUT CODE. Per beat slot = onehot(pitch)[12] + [attack] = 13 dims
   (attack as its own channel), so INPUT_DIM = 3*4*13 = 156. The brief's "12 + 1"
   encoding. The old 24-dim sustained+onset code is available for comparison
   (knob: input_mode = "attack" | "dual" | "factored").

Learning rule (local, Dale-consistent)
--------------------------------------
The decoder uses a signed error e_o = target_o - prediction_o (the signed-XOR
motif) but applied on an E/I substrate:

    dW_out[o, j]  proportional to  e_o * sign_j * h_j        (then clip to >= 0)

so the *direction* of every weight change is e_o * sign_j: an excitatory active
unit is potentiated when the target needs more drive and depressed when it has
too much; an inhibitory active unit does the opposite (it is recruited to
suppress a spuriously-firing output). Clipping to [floor, W_MAX] keeps Dale's
law. The encoder learns by pure Hebbian LTP (post * pre), or is a frozen random
projection (flag: encoder_learn).

Run
---
    python biomodel2.py                 # train full model, evaluate, save summary
    python biomodel2.py --epochs 60 --latent 256
    python biomodel2.py --ablation      # leave-one-out study over every mechanism
    python biomodel2.py --quick         # fast smoke test
"""

import argparse
import json
from dataclasses import dataclass, replace as _dc_replace
from contextlib import contextmanager

import numpy as np

from corpus import (
    BAR_SIZE, N_PITCHES, load_corpus, flatten, names_str,
    TRAIN_PATH, NOVEL_PATH, RESULTS_DIR,
)


# ============================================================
# 1. Dimensions and constants
# ============================================================

N_NOTES   = N_PITCHES        # 12 chromatic pitches
K_BAR     = 3                # context bars
OUTPUT_SLOT = N_NOTES + 1    # 13: 12 pitch + 1 attack

LATENT    = 256
EPOCHS    = 120
LR_DEC    = 0.04             # decoder learning rate
LR_ENC    = 0.02             # encoder (Hebbian) learning rate

# k-WTA: active fraction of the latent code.
KWTA_FRAC = 0.15             # 256 -> ~38 active (requested 30-50 band)

# Dale's-law population.
INH_FRAC  = 0.20             # fraction of latent units that are inhibitory

# Input wiring sparsity.
CONN_DENSITY = 0.10          # ~10% of possible input synapses kept per unit

# Multi-synapse / quantal weights.
N_SYN_MAX = 10               # up to 10 contacts per connection ...
SYN_W_MAX = 1.0              # ... each able to grow to strength 1.0
W_MAX     = N_SYN_MAX * SYN_W_MAX           # effective ceiling = 10
CONSOLIDATE_REPS = 3         # repetitions to lock a freshly formed contact

DIV_INH_EPS = 1e-2

# Novelty-gated plasticity (experiment 1). When NOVELTY_GATE is True, each
# window's weight update is scaled by a sigmoid of its prediction error, so
# already-known windows stop being rewritten ("learn on surprise").
NOVELTY_GATE = False
NOVELTY_THR  = 0.05      # surprise level (mean RPE) at which the gate is half-open
NOVELTY_TEMP = 0.02      # softness of the gate (smaller = sharper on/off)


# ============================================================
# 2. Ablation switchboard
# ============================================================
#
# Each biological mechanism is a single flag. Defaults = the full model. The
# leave-one-out driver (run_ablation / --ablation) flips one at a time.
#
#   dale          True  -> E/I neurons, non-negative weights, sign carried by
#                          presynaptic type.   False -> free signed weights
#                          (the old biomodel.py substrate).
#   multisyn      True  -> quantal bundle of up to N_SYN_MAX contacts; LTP fills
#                          a contact before recruiting the next.   False -> a
#                          single continuous synapse (still capped at W_MAX).
#   consolidation True  -> consolidated contacts set an LTD-proof floor.
#                          False -> no protection; LTD can fully reverse LTP.
#   ltd           True  -> depression allowed (signed error).   False -> LTP only
#                          (unsigned |error|): isolates the value of the sign /
#                          depression channel.
#   kwta          True  -> top-(KWTA_FRAC) sparse code.   False -> dense code.
#   sparse_wiring True  -> fixed random conn_density input mask.   False -> dense
#                          input wiring (every input synapse present).
#   encoder_learn False -> FIXED random projection (fly-KC style expansion;
#                          the full model). True -> homeostatic Hebbian LTP on
#                          W_in. (Learning the expansion layer collapses the
#                          code via winner-take-all runaway and underperforms
#                          the random projection here -- it is kept only as an
#                          informative ablation, not part of the full model.)
#   div_inh       True  -> divisive inhibition (I-pool normalization) before
#                          k-WTA.   False -> no normalization.
#   classifier    True  -> song-ID recognition head (also drives a familiarity
#                          term).   False -> no recognition head.
# ============================================================

@dataclass
class AblationConfig:
    dale:          bool  = True
    multisyn:      bool  = True
    consolidation: bool  = True
    ltd:           bool  = True
    kwta:          bool  = True
    sparse_wiring: bool  = True
    encoder_learn: bool  = False       # full model = FIXED random projection
    div_inh:       bool  = True
    classifier:    bool  = True
    conn_density:  float = CONN_DENSITY
    input_mode:    str   = "factored"      # "factored" (14, paper default) | "attack" (13) | "dual" (24)

    def slot_dim(self):
        if self.input_mode == "attack":   return OUTPUT_SLOT      # 13: pitch[12] + attack bit
        if self.input_mode == "dual":     return 2 * N_NOTES      # 24
        if self.input_mode == "factored": return N_NOTES + 2      # 14: pitch[12] + 2-way attack one-hot
        raise ValueError(self.input_mode)

    def out_dim(self):
        # decoder output slot dim; for factored it matches the input slot (format-closed)
        return N_NOTES + 2 if self.input_mode == "factored" else OUTPUT_SLOT

    def input_dim(self):
        return K_BAR * BAR_SIZE * self.slot_dim()

    def label(self):
        full = AblationConfig()
        diffs = []
        for f in ("dale", "multisyn", "consolidation", "ltd", "kwta",
                  "sparse_wiring", "encoder_learn", "div_inh", "classifier"):
            if getattr(self, f) != getattr(full, f):
                diffs.append(f"{f}={getattr(self, f)}")
        if self.conn_density != full.conn_density:
            diffs.append(f"conn_density={self.conn_density}")
        if self.input_mode != full.input_mode:
            diffs.append(f"input_mode={self.input_mode}")
        return "full" if not diffs else ", ".join(diffs)


CFG = AblationConfig()


def set_ablation(cfg=None, **overrides):
    global CFG
    prev = CFG
    if cfg is None:
        cfg = _dc_replace(CFG, **overrides) if overrides else AblationConfig()
    elif overrides:
        cfg = _dc_replace(cfg, **overrides)
    CFG = cfg
    return prev


@contextmanager
def ablation_context(cfg=None, **overrides):
    prev = set_ablation(cfg, **overrides)
    try:
        yield CFG
    finally:
        set_ablation(prev)


# ============================================================
# 3. Helpers
# ============================================================

def onehot(i, n):
    v = np.zeros(n)
    v[i] = 1.0
    return v


def bars_to_vec(bars):
    """K_BAR bars of [(pitch, attack) x 4] -> input vector.

    input_mode "attack": per slot = onehot(pitch)[12] + [attack]   -> 13 dims
    input_mode "dual"    : per slot = sustained[12] + onset[12]       -> 24 dims
    input_mode "factored": per slot = onehot(pitch)[12] + onehot(attack)[2] -> 14 dims
                           (decoder emits the same 14-d code: format-closed)
    """
    vecs = []
    mode = CFG.input_mode
    for bar in bars:
        for pitch, attack in bar:
            p = onehot(pitch, N_NOTES)
            if mode == "dual":
                vecs.append(np.concatenate([p, p * float(attack)]))
            elif mode == "factored":
                vecs.append(np.concatenate([p, onehot(int(attack), 2)]))  # pitch + {sustain,onset}
            else:  # attack
                vecs.append(np.concatenate([p, [float(attack)]]))
    return np.concatenate(vecs)


def wta(v):
    out = np.zeros_like(v)
    out[np.argmax(v)] = 1.0
    return out


def k_wta(act, k, alive):
    """Top-k by activation among alive units; selected keep their value."""
    out = np.zeros_like(act)
    masked = np.where(alive, act, -np.inf)
    k_eff = min(k, int(alive.sum()))
    if k_eff > 0:
        idx = np.argpartition(masked, -k_eff)[-k_eff:]
        out[idx] = act[idx]
    return out


# ============================================================
# 4. Synapse bundles (multi-synapse quantal weights)
# ============================================================
#
# A weight matrix is stored as three same-shaped arrays:
#   W       effective weight in [0, W_MAX]   (sum of contact strengths)
#   consol  consolidated floor               (locked contacts; LTD-proof)
#   reps    repetition counter toward locking the next contact
#
# potentiate() and depress() are masked, vectorised, and Dale-aware. They are
# the ONLY way weights change, so every mechanism flag lives in one place.

def new_bundle(shape, mask=None):
    return {
        "W":      np.zeros(shape),
        "consol": np.zeros(shape),
        "reps":   np.zeros(shape, dtype=np.int32),
        "mask":   np.ones(shape) if mask is None else mask,
    }


def bundle_update(b, delta, lr):
    """Apply a signed weight-change field `delta` (post x pre) with learning
    rate `lr`. Potentiation and depression are split so consolidation,
    quantisation, and the Dale floor all apply correctly.

    `delta` has the sign of the desired change; under Dale the caller has
    already folded in the presynaptic sign, so positive delta == LTP.
    """
    W, consol, reps, mask = b["W"], b["consol"], b["reps"], b["mask"]
    dpos = np.maximum(delta, 0.0) * lr
    dneg = np.maximum(-delta, 0.0) * lr           # magnitude of depression

    if not CFG.ltd:
        dneg[:] = 0.0                             # LTP-only ablation

    # ---- potentiation (LTP) ----
    ltp_hit = dpos > 0
    if ltp_hit.any():
        before_floor = np.floor(W)
        W += dpos
        np.clip(W, 0.0, W_MAX, out=W)
        if CFG.multisyn and CFG.consolidation:
            # a contact just (further) formed: count a repetition; lock when a
            # new integer contact has survived CONSOLIDATE_REPS potentiations.
            reps[ltp_hit] += 1
            new_contact = (np.floor(W) > consol) & (reps >= CONSOLIDATE_REPS) & ltp_hit
            consol[new_contact] = np.floor(W[new_contact])
            reps[new_contact] = 0
        elif CFG.consolidation:
            # continuous (non-quantal) consolidation: a stiffness-like floor that
            # creeps up toward the current weight as potentiation accumulates.
            consol += 0.25 * dpos * (W - consol > 0)
            np.minimum(consol, W, out=consol)

    # ---- depression (LTD) ----
    ltd_hit = dneg > 0
    if ltd_hit.any():
        floor = consol if CFG.consolidation else (np.zeros_like(W) if CFG.dale
                                                  else np.full_like(W, -W_MAX))
        W -= dneg
        np.maximum(W, floor, out=W)
        if not CFG.dale:
            np.clip(W, -W_MAX, W_MAX, out=W)

    W *= mask                                     # absent synapses stay absent


def effective(b):
    """Effective weight matrix actually used in forward sums."""
    return b["W"] * b["mask"]


# ============================================================
# 5. Model
# ============================================================

def _sparse_mask(n_rows, fan_in, density):
    m = int(round(max(0.0, min(1.0, density)) * fan_in))
    mask = np.zeros((n_rows, fan_in))
    if m >= fan_in:
        mask[:] = 1.0
        return mask
    m = max(1, m)
    for r in range(n_rows):
        mask[r, np.random.choice(fan_in, m, replace=False)] = 1.0
    return mask


def init_model(n_songs):
    in_dim = CFG.input_dim()
    N = LATENT

    # Dale population: assign neuron types and their outgoing sign.
    if CFG.dale:
        n_inh = int(round(INH_FRAC * N))
        types = np.array(["E"] * (N - n_inh) + ["I"] * n_inh)
        np.random.shuffle(types)
        sign = np.where(types == "E", 1.0, -1.0)
    else:
        types = np.array(["E"] * N)               # no sign constraint
        sign = np.ones(N)

    m = {"sign": sign, "types": types, "in_dim": in_dim, "n_songs": n_songs}

    # Input wiring mask (~10% if sparse_wiring).
    if CFG.sparse_wiring and CFG.conn_density < 1.0:
        in_mask = _sparse_mask(N, in_dim, CFG.conn_density)
    else:
        in_mask = np.ones((N, in_dim))

    # Encoder bundle. Small positive init so the random projection already
    # separates contexts before any learning.
    m["W_in"] = new_bundle((N, in_dim), mask=in_mask)
    init = np.abs(np.random.randn(N, in_dim)) * (0.5 / np.sqrt(max(1, in_dim * CFG.conn_density)))
    if not CFG.dale:
        init = np.random.randn(N, in_dim) * (0.5 / np.sqrt(max(1, in_dim * CFG.conn_density)))
    m["W_in"]["W"] = init * in_mask

    # Decoder bundles: one head per beat slot of the predicted bar.
    od = CFG.out_dim()
    for i in range(BAR_SIZE):
        b = new_bundle((od, N))
        b["W"] = (np.abs(np.random.randn(od, N)) if CFG.dale
                  else np.random.randn(od, N)) * (0.3 / np.sqrt(N))
        m[f"W_out{i}"] = b

    # Recognition (song-ID) head.
    if CFG.classifier:
        b = new_bundle((n_songs, N))
        b["W"] = (np.abs(np.random.randn(n_songs, N)) if CFG.dale
                  else np.random.randn(n_songs, N)) * (0.3 / np.sqrt(N))
        m["W_cls"] = b

    m["alive"] = np.ones(N, dtype=bool)
    return m


def encode(model, ctx_bars):
    """3 context bars -> sparse latent code h (>= 0) and signed code (sign*h)."""
    x = bars_to_vec(ctx_bars)
    pre = effective(model["W_in"]) @ x            # >= 0 under Dale (x, W >= 0)
    if CFG.div_inh:
        pre = pre / (DIV_INH_EPS + pre.mean())
    act = np.maximum(pre, 0.0)
    if CFG.kwta:
        k = max(1, int(round(KWTA_FRAC * model["alive"].sum())))
        h = k_wta(act, k, model["alive"])
    else:
        h = act * model["alive"]
    return x, h


def signed_code(model, h):
    return model["sign"] * h                      # E:+h  I:-h  (Dale)


def predict_bar(model, h):
    sh = signed_code(model, h)
    factored = (CFG.input_mode == "factored")
    preds = []
    for i in range(BAR_SIZE):
        drive = effective(model[f"W_out{i}"]) @ sh
        pitch_logits = drive[:N_NOTES]
        if factored:
            attack_logits = drive[N_NOTES:N_NOTES + 2]
            attack_state = int(np.argmax(attack_logits))         # 0=sustain, 1=onset
            preds.append((wta(pitch_logits), attack_state, pitch_logits, attack_logits))
        else:
            attack_logit = drive[N_NOTES]
            preds.append((wta(pitch_logits), int(attack_logit > 0.5),
                          pitch_logits, attack_logit))
    return preds


def classify(model, h):
    if not CFG.classifier:
        return None
    return effective(model["W_cls"]) @ signed_code(model, h)


# ============================================================
# 6. Training on one song
# ============================================================

def train_on_song(model, bars, song_id):
    cp = ca = ts = cc = tc = 0
    rpe_sum = 0.0
    rpe_n = 0
    sh_cache = None

    for t in range(K_BAR, len(bars)):
        ctx = bars[t - K_BAR: t]
        tgt = bars[t]
        x, h = encode(model, ctx)
        sh = signed_code(model, h)
        preds = predict_bar(model, h)

        # ---- novelty gate: scale plasticity by this window's surprise ----
        # gate = 1 normally; with NOVELTY_GATE on, gate is driven by the window's
        # prediction error (RPE): low surprise (already known) -> ~0 (frozen),
        # high surprise (novel) -> ~1. This is the "learn on novelty" motif and
        # also self-freezes near-correct windows so late updates stop perturbing
        # the shared code (the AR-dip cause).
        win_err = 0.0
        for i in range(BAR_SIZE):
            tp, ta = tgt[i]
            pp, pa, _, _ = preds[i]
            win_err += (np.abs(onehot(tp, N_NOTES) - pp).mean()
                        + abs(float(ta) - float(pa))) / 2.0
        win_err /= BAR_SIZE
        if NOVELTY_GATE:
            # smooth gate: ~0 below threshold, ramps to 1 with surprise
            gate = 1.0 / (1.0 + np.exp(-(win_err - NOVELTY_THR) / NOVELTY_TEMP))
        else:
            gate = 1.0

        # ---- decoder: signed local rule on an E/I substrate ----
        rpe_step = 0.0
        for i in range(BAR_SIZE):
            tgt_pitch, tgt_attack = tgt[i]
            pred_pitch, pred_attack, _, _ = preds[i]
            e_pitch  = onehot(tgt_pitch, N_NOTES) - pred_pitch        # signed-XOR
            if CFG.input_mode == "factored":
                e_attack = onehot(int(tgt_attack), 2) - onehot(int(pred_attack), 2)
                e = np.concatenate([e_pitch, e_attack])               # (14,)
            else:
                e = np.concatenate([e_pitch, [float(tgt_attack) - float(pred_attack)]])  # (13,)
            rpe_step += np.abs(e).mean()

            # dW[o,j] ~ e_o * sign_j * h_j   (presynaptic sign folded in via sh)
            delta = np.outer(e, sh)
            bundle_update(model[f"W_out{i}"], delta, LR_DEC * gate)

            if int(np.argmax(pred_pitch)) == tgt_pitch:
                cp += 1
            if pred_attack == tgt_attack:
                ca += 1
            ts += 1
        rpe_sum += rpe_step / BAR_SIZE
        rpe_n += 1

        # ---- recognition head (same signed rule) ----
        if CFG.classifier:
            cls = classify(model, h)
            cls_wta = wta(cls)
            e_cls = onehot(song_id, model["n_songs"]) - cls_wta
            bundle_update(model["W_cls"], np.outer(e_cls, sh), LR_DEC * gate)
            if int(np.argmax(cls)) == song_id:
                cc += 1
            tc += 1

        # ---- encoder: optional homeostatic Hebbian LTP (post * pre) ----
        # Off in the full model. When on, each row is renormalised to a fixed
        # L1 budget after potentiation so a few units cannot monopolise the
        # code (the runaway that makes a naive learned encoder collapse).
        if CFG.encoder_learn:
            delta_in = np.outer(h, x)             # >= 0 -> LTP only
            bundle_update(model["W_in"], delta_in, LR_ENC * gate)
            W = model["W_in"]["W"]; mask = model["W_in"]["mask"]
            rs = (W * mask).sum(1, keepdims=True); rs[rs == 0] = 1.0
            budget = mask.sum(1, keepdims=True) * 0.05
            model["W_in"]["W"] = W * (budget / rs)

    return cp, ca, ts, cc, tc, (rpe_sum / max(1, rpe_n))


def train_model(songs_bars, epochs=EPOCHS, verbose=True, log_every=10, ar_every=0):
    model = init_model(len(songs_bars))
    n = len(songs_bars)
    acc_hist, cls_hist, rpe_hist = [], [], []
    ar_hist = []                                       # (epoch, tf_acc, ar_acc)
    for ep in range(epochs):
        cp = ca = ts = cc = tc = 0
        rpe = 0.0
        for sid in np.random.permutation(n):
            r = train_on_song(model, songs_bars[sid], sid)
            cp += r[0]; ca += r[1]; ts += r[2]; cc += r[3]; tc += r[4]; rpe += r[5]
        acc = cp / ts if ts else 0.0
        cls = cc / tc if tc else 0.0
        rpe /= n
        acc_hist.append(acc); cls_hist.append(cls); rpe_hist.append(rpe)
        if verbose and (ep % log_every == 0 or ep == epochs - 1):
            dec = model["W_out0"]
            wmean = effective(dec).mean()
            cons = dec["consol"].mean()
            nsyn = np.ceil(effective(dec)).mean()      # mean formed contacts
            print(f"  ep {ep:03d}  acc_pitch={acc:.3f}  recog={cls:.3f}  "
                  f"rpe={rpe:.3f}  Wout_mean={wmean:.3f}  consol={cons:.3f}  "
                  f"syn/conn={nsyn:.2f}")
        # AR checkpoint: measure (teacher-forced and) autoregressive recall mid-training
        if ar_every and (ep % ar_every == 0 or ep == epochs - 1):
            tf_ck, ar_ck = evaluate(model, songs_bars)
            ar_hist.append((ep, float(tf_ck.mean()), float(ar_ck.mean())))
            if verbose:
                print(f"      [AR checkpoint] ep {ep:03d}: "
                      f"TF={tf_ck.mean():.4f}  AR={ar_ck.mean():.4f}")
    return model, {"acc": acc_hist, "recog": cls_hist, "rpe": rpe_hist,
                   "ar_checkpoints": ar_hist}


# ============================================================
# 7. Evaluation
# ============================================================

def evaluate(model, songs_bars):
    """Teacher-forced and autoregressive pitch accuracy (means over songs)."""
    tf, ar = [], []
    for bars in songs_bars:
        # teacher forced
        cp = ts = 0
        for t in range(K_BAR, len(bars)):
            _, h = encode(model, bars[t - K_BAR: t])
            pred = predict_bar(model, h)
            for i in range(BAR_SIZE):
                if int(np.argmax(pred[i][0])) == bars[t][i][0]:
                    cp += 1
                ts += 1
        tf.append(cp / ts if ts else 0.0)
        # autoregressive
        ctx = [list(b) for b in bars[:K_BAR]]
        cp = ts = 0
        for t in range(K_BAR, len(bars)):
            _, h = encode(model, ctx[-K_BAR:])
            pred = predict_bar(model, h)
            gen = [(int(np.argmax(pred[i][0])), pred[i][1]) for i in range(BAR_SIZE)]
            for i in range(BAR_SIZE):
                if gen[i][0] == bars[t][i][0]:
                    cp += 1
                ts += 1
            ctx.append(gen)
        ar.append(cp / ts if ts else 0.0)
    return np.array(tf), np.array(ar)


def recognition_accuracy(model, songs_bars):
    if not CFG.classifier:
        return None
    correct = 0
    for sid, bars in enumerate(songs_bars):
        votes = np.zeros(model["n_songs"])
        for t in range(K_BAR, len(bars)):
            _, h = encode(model, bars[t - K_BAR: t])
            votes += classify(model, h)
        if int(np.argmax(votes)) == sid:
            correct += 1
    return correct / len(songs_bars)


def song_rpe(model, bars):
    """Mean teacher-forced prediction-error magnitude over a song (novelty)."""
    tot = 0.0
    n = 0
    for t in range(K_BAR, len(bars)):
        _, h = encode(model, bars[t - K_BAR: t])
        preds = predict_bar(model, h)
        e = 0.0
        for i in range(BAR_SIZE):
            tgt_pitch, _ = bars[t][i]
            e += np.abs(onehot(tgt_pitch, N_NOTES) - preds[i][0]).mean()
        tot += e / BAR_SIZE
        n += 1
    return tot / max(1, n)


def novelty_report(model, train_bars, novel_songs):
    """Calibrate a known-vs-novel threshold on training RPE and apply it."""
    known = np.array([song_rpe(model, b) for b in train_bars])
    thr = known.mean() + 2 * known.std()
    rows = []
    for name, bars in novel_songs:
        r = song_rpe(model, bars)
        rows.append({"name": name, "rpe": float(r), "verdict":
                     "novel" if r > thr else "known"})
    detected = sum(rw["verdict"] == "novel" for rw in rows)
    return {
        "threshold": float(thr),
        "known_rpe_mean": float(known.mean()),
        "known_rpe_std": float(known.std()),
        "novel_detected": detected,
        "n_novel": len(novel_songs),
        "rows": rows,
    }


# ============================================================
# 8. Pipeline
# ============================================================

def run(epochs=EPOCHS, latent=None, seed=0, verbose=True, novelty=True, ar_every=0):
    global LATENT
    if latent is not None:
        LATENT = latent
    np.random.seed(seed)

    train = load_corpus(TRAIN_PATH)
    novel = load_corpus(NOVEL_PATH)
    train_bars = [b for _, b in train]

    if verbose:
        n_inh = int(round(INH_FRAC * LATENT)) if CFG.dale else 0
        k = int(round(KWTA_FRAC * LATENT))
        print(f"[{CFG.label()}]")
        print(f"  latent={LATENT} (E={LATENT-n_inh}, I={n_inh})  "
              f"k-WTA={k}  input_dim={CFG.input_dim()}  "
              f"conn_density={CFG.conn_density if CFG.sparse_wiring else 1.0}")

    model, hist = train_model(train_bars, epochs=epochs, verbose=verbose,
                              ar_every=ar_every)

    tf, ar = evaluate(model, train_bars)
    recog = recognition_accuracy(model, train_bars)
    out = {
        "ablation": CFG.label(),
        "epochs": epochs, "latent": LATENT, "seed": seed,
        "tf_pitch_acc": float(tf.mean()),
        "ar_pitch_acc": float(ar.mean()),
        "recognition_acc": (None if recog is None else float(recog)),
        "final_train_acc": hist["acc"][-1],
    }
    if novelty:
        out["novelty"] = novelty_report(model, train_bars, novel)
    if hist.get("ar_checkpoints"):
        out["ar_checkpoints"] = hist["ar_checkpoints"]
    if verbose:
        print(f"  => TF pitch acc = {tf.mean():.3f}   AR pitch acc = {ar.mean():.3f}"
              + ("" if recog is None else f"   recognition = {recog:.3f}"))
        if novelty:
            nv = out["novelty"]
            print(f"  => novelty: detected {nv['novel_detected']}/{nv['n_novel']} "
                  f"held-out songs (thr={nv['threshold']:.3f})")
    return model, hist, out


# ============================================================
# 9. Leave-one-out ablation study
# ============================================================

def run_ablation(epochs=120, seed=0):
    # mechanisms that are ON in the full model (leave-one-out turns each OFF)
    mechanisms = ["dale", "multisyn", "consolidation", "ltd", "kwta",
                  "sparse_wiring", "div_inh", "classifier"]
    results = []
    # full model
    set_ablation(AblationConfig())
    print("\n=== FULL MODEL ===")
    _, _, full = run(epochs=epochs, seed=seed, verbose=True, novelty=True)
    results.append(full)
    # leave-one-out
    for mech in mechanisms:
        with ablation_context(**{mech: False}):
            print(f"\n=== ABLATE {mech} ===")
            _, _, r = run(epochs=epochs, seed=seed, verbose=True, novelty=True)
            results.append(r)
    # additive probe: turn the (off-by-default) Hebbian encoder ON
    with ablation_context(encoder_learn=True):
        print("\n=== +encoder_learn (Hebbian expansion) ===")
        _, _, r = run(epochs=epochs, seed=seed, verbose=True, novelty=True)
        results.append(r)
    # table
    print("\n" + "=" * 78)
    print(f"{'configuration':<26}{'TF acc':>9}{'AR acc':>9}{'recog':>9}{'novelty':>12}")
    print("-" * 78)
    for r in results:
        nv = r.get("novelty") or {}
        nov = (f"{nv.get('novel_detected','-')}/{nv.get('n_novel','-')}"
               if nv else "-")
        rec = "-" if r["recognition_acc"] is None else f"{r['recognition_acc']:.3f}"
        print(f"{r['ablation']:<26}{r['tf_pitch_acc']:>9.3f}"
              f"{r['ar_pitch_acc']:>9.3f}{rec:>9}{nov:>12}")
    print("=" * 78)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / "ablation_biomodel2.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"saved {out}")
    return results


# ============================================================
# 10. CLI
# ============================================================

def _cli(argv=None):
    ap = argparse.ArgumentParser(description="Dale's-law multi-synapse memorizer")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--latent", type=int, default=LATENT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", action="store_true",
                    help="run the leave-one-out ablation study")
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke test (few epochs, small latent)")
    ap.add_argument("--input-mode", choices=["attack", "dual"], default="attack")
    ap.add_argument("--ar-every", type=int, default=0,
                    help="evaluate autoregressive recall every N epochs (0=off, "
                         "only at end). Use to see the AR trajectory during training.")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _cli()
    set_ablation(input_mode=args.input_mode)
    if args.quick:
        run(epochs=15, latent=128, seed=args.seed)
    elif args.ablation:
        run_ablation(epochs=args.epochs if args.epochs != EPOCHS else 120,
                     seed=args.seed)
    else:
        model, hist, summary = run(epochs=args.epochs, latent=args.latent,
                                   seed=args.seed, ar_every=args.ar_every)
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(RESULTS_DIR / "biomodel2_summary.json", "w",
                  encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nsaved {RESULTS_DIR / 'biomodel2_summary.json'}")
    print("Done.")
