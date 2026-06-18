"""
Corpus loading and bar-string parsing for the 50-melody dataset.

The dataset lives as JSON under data/. Each melody is a {"name", "bars"}
object; each bar is a space-separated token string. See README.md for the
notation in full; the short version:

    "C"      -> C quarter note (1 beat, attack=1)
    "C2"     -> C half note (2 beats: attack=1, then sustain attack=0)
    "C3"     -> C dotted half (3 beats)
    "C4"     -> C whole note (4 beats)
    "C#"     -> C sharp (also "C^")
    "_C"     -> C flat (also "Cb")
    "C C"    -> two onset quarter notes (two attacks, not the same as "C2")

A bar contains BAR_SIZE = 4 slots. Each slot is a (pitch, attack) tuple.
"""

import json
from pathlib import Path

BAR_SIZE = 4
N_PITCHES = 12
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Default paths (relative to this file)
_REPO_DIR = Path(__file__).resolve().parent
DATA_DIR = _REPO_DIR / "data"
RESULTS_DIR = _REPO_DIR / "results"
TRAIN_PATH = DATA_DIR / "songs_train.json"
NOVEL_PATH = DATA_DIR / "songs_novel.json"


def parse_bar_str(s, beats=BAR_SIZE):
    """Parse one bar string into a list of (pitch, attack) tuples.

    Rule: the first slot of each token has attack=1 (onset); the rest are
    attack=0 (sustain / continuation of the same pitch).

    Example:  "A A G2"   -> [(9,1),(9,1),(7,1),(7,0)]
              "G2 C C"   -> [(7,1),(7,0),(0,1),(0,1)]
              "C4"       -> [(0,1),(0,0),(0,0),(0,0)]
    """
    slots = []
    for tok in s.split():
        tok = tok.strip()
        if not tok:
            continue
        # Duration suffix: trailing digit
        dur = 1
        if tok and tok[-1].isdigit():
            dur = int(tok[-1])
            tok = tok[:-1]
        # Accidental prefix _ (flat)
        acc = 0
        if tok.startswith("_"):
            acc = -1
            tok = tok[1:]
        # Accidental suffix ^ or # (sharp); or b (flat, but careful not to
        # eat the B natural)
        if tok.endswith("^") or tok.endswith("#"):
            acc = +1
            tok = tok[:-1]
        elif tok.endswith("b") and len(tok) > 1:
            acc = -1
            tok = tok[:-1]
        pitch = (_NOTE_BASE[tok.upper()] + acc) % 12
        slots.append((pitch, 1))                # onset
        for _ in range(dur - 1):
            slots.append((pitch, 0))            # sustain
    if len(slots) != beats:
        raise ValueError(
            f"Bar {s!r} produced {len(slots)} slots, expected {beats}"
        )
    return slots


def load_corpus(path=TRAIN_PATH):
    """Load a JSON corpus and parse every bar.

    Returns a list of (name, bars_parsed) tuples, where bars_parsed is a list
    of bars and each bar is a list of (pitch, attack) tuples.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    songs = data["songs"] if isinstance(data, dict) else data
    return [(s["name"], [parse_bar_str(b) for b in s["bars"]]) for s in songs]


def load_corpus_raw(path=TRAIN_PATH):
    """Load a JSON corpus and return raw bar strings (no parsing).

    Useful when you want to display the original notation alongside parsed
    output (e.g. in the playback CLI).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    songs = data["songs"] if isinstance(data, dict) else data
    return [(s["name"], list(s["bars"])) for s in songs]


def names_str(slots, show_cont=True):
    """Render a flat list of (pitch, attack) as a readable string.

    Sustained slots are prefixed with ~ when show_cont is True, so "G G" and
    "G2" look distinct in the output: 'G G' vs 'G ~G'.
    """
    parts = []
    for p, a in slots:
        name = NOTE_NAMES[p]
        if a == 0 and show_cont:
            name = "~" + name
        parts.append(name)
    return " ".join(parts)


def flatten(bars):
    """Flatten a list-of-bars (each a list of slots) to a flat slot list."""
    return [s for bar in bars for s in bar]


def safe_filename(name):
    """Sanitize a song title so it can be used as a filename component."""
    keep = "-_.() "
    out = "".join(c if c.isalnum() or c in keep else "_" for c in name)
    return out.strip().replace(" ", "_")


if __name__ == "__main__":
    train = load_corpus(TRAIN_PATH)
    novel = load_corpus(NOVEL_PATH)
    print(f"train: {len(train)} songs, "
          f"{sum(len(b) for _, b in train)} bars, "
          f"{sum(len(b)*BAR_SIZE for _, b in train)} slots")
    print(f"novel: {len(novel)} songs")
    print(f"\nfirst train song: {train[0][0]!r}")
    print(f"  bars parsed: {len(train[0][1])}")
    print(f"  first 2 bars: {train[0][1][:2]}")
