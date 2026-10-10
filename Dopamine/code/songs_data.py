"""Children's-song corpus (Pena Fernandez, Lloret Iglesias & Marco de Lucas, musicmem-engrammer repository).
Context = 3 bars x 4 beats x (12 pitch + 2 attack) = 168 bits; target = next bar (56 bits).
PITCH_ONLY=True zeroes the attack bits of the context (the target keeps them).
Set MUSICMEM=/path/to/musicmem-engrammer (default: the repository root two levels up, where corpus.py and data/ live)."""
import os, sys, numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))   # musicmem-engrammer repository root (corpus.py, data/)
REPO = os.environ.get('MUSICMEM', _ROOT if os.path.exists(os.path.join(_ROOT, 'corpus.py')) else os.path.join(_HERE, 'musicmem'))
sys.path.insert(0, REPO)
from corpus import load_corpus, TRAIN_PATH, NOVEL_PATH

PITCH_ONLY = True


def slot_vec(p, a):
    v = np.zeros(14); v[p] = 1; v[12 + a] = 1; return v


def windows(bars, K=3):
    X, Y = [], []
    for t in range(K, len(bars)):
        v = np.concatenate([slot_vec(p, a) for bar in bars[t - K:t] for (p, a) in bar])
        if PITCH_ONLY: v = (v.reshape(12, 14) * np.r_[np.ones(12), 0, 0][None, :]).ravel()
        X.append(v); Y.append(np.concatenate([slot_vec(p, a) for (p, a) in bars[t]]))
    return np.array(X), np.array(Y)


def bars_to_context(bars):
    """Context vector from three bars (lists of (pitch, attack)), with the same coding as windows()."""
    v = np.concatenate([slot_vec(p, a) for bar in bars for (p, a) in bar])
    if PITCH_ONLY: v = (v.reshape(12, 14) * np.r_[np.ones(12), 0, 0][None, :]).ravel()
    return v


def load_bars():
    return [b for _, b in load_corpus(TRAIN_PATH)]


_cache = {}
def load():
    if PITCH_ONLY not in _cache:
        _cache[PITCH_ONLY] = ([windows(b) for _, b in load_corpus(TRAIN_PATH)], [windows(b) for _, b in load_corpus(NOVEL_PATH)])
    return _cache[PITCH_ONLY]


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    return float(np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :]))
