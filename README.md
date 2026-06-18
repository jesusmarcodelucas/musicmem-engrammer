# musicmem-engrammer

A **Dale's-law-compliant, gradient-free sparse associative memory** for short
symbolic sequences, built on the ENGRAMMER signed-XOR motif. The task is a controlled
sequence-memory benchmark: memorize and recall 50 short monophonic children's melodies
in 4/4, and flag 12 held-out melodies as novel.

The network is a frozen sparse random encoder (mushroom-body-like, \~10% wiring) read
out by a local **signed-XOR** rule on **explicit multi-contact (quantal) synapses**,
under **Dale's law** (separate excitatory/inhibitory populations, non-negative weights).
There is no backpropagation, no weight transport, and no learned front end.

## Headline result

With the **factored** input encoding (each beat = a 12-way pitch one-hot **and** a
separate 2-way {onset, sustain} one-hot, 14-d; the decoder emits the same 14-d code, so
the autoregressive loop closes in a single neural format), the model:

* stores the 50-melody training corpus **exactly**;
* reaches **100% teacher-forced and autoregressive recall at L=512** (all seeds);
* recognizes all 50 training melodies and flags all 12 held-out melodies as novel with
zero overlap.

Teacher-forced storage saturates between L=128 and L=256; perfect free-running recall
arrives by L=512. A size/onset-matched **random** corpus is recalled at least as well at
every capacity, so musical structure does not aid recall here — the system is a memory,
not a generalizer.

## Repository layout

```
biomodel2.py            the model (Dale's-law, signed-XOR, multi-synapse, k-WTA).
                        Default input\_mode="factored" (paper). Also supports
                        "attack" (13-d) and "dual" (24-d) for the encoding comparison.
corpus.py               loads data/songs\_\*.json into (pitch, attack) bar sequences.
probe\_basins.py         real-vs-random corpus helpers.
run\_sweep.py            capacity sweep (Table 2).
run\_ablation\_table.py   leave-one-out ablation, L=256 (Table 1).
run\_wiring.py           input-wiring-density sweep, L=256 (Table 3).
run\_random.py           real vs matched-random corpus (Table 4).
run\_encoding\_compare.py attack vs dual vs factored (Table 5 / Appendix).
run\_L1024.py            single large-capacity run.
run\_novelty.py          novelty / known-vs-novel probing.
make\_figures.py         regenerates the figures from results/\*.json.
data/                   songs\_train.json (50), songs\_novel.json (12, incl. El cocherito leré).
results/                authoritative result JSONs behind every table and figure.
```

## Requirements

Python 3 with `numpy` and `matplotlib`. No GPU; everything runs on a laptop CPU.

```bash
pip install numpy matplotlib
```

## Reproduce

All scripts default to the factored encoding (the paper model).

```bash
python run\_sweep.py            --seeds 0 1 2     # capacity (Table 2)
python run\_ablation\_table.py   --seed 0          # ablation, L=256 (Table 1; run 3 seeds, aggregate)
python run\_wiring.py           --seed 0          # wiring, L=256 (Table 3; run 3 seeds, aggregate)
python run\_random.py                             # real vs random (Table 4)
python run\_encoding\_compare.py --seeds 5         # attack/dual/factored (Table 5)
python make\_figures.py                           # regenerate figures into results/
```

To run an alternative encoding from a script:
`B.set\_ablation(B.AblationConfig(input\_mode="attack"))`.

Capacity-sweep sparsity: the paper uses k = 8, 15, 30, 60, 60 for
L = 64, 128, 256, 512, 1024 (k/L ≈ 0.117, capped at 60).

## Paper

The preprint (PDF) is included in this repository: *Signed-XOR Error and Sparse Coding
in a Dale-Compliant Substrate for Sequence Memorization* (Peña Fernández, Lloret Iglesias,
Marco de Lucas).

## Acknowledgments

Supported by the ENGRAMMER project (IASOMM2024007), funded by the EU NextGeneration
programme under the Recovery and Resilience Facility (RRF).

