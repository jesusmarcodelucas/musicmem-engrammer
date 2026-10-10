#!/usr/bin/env bash
# ENGRAMMER-DOPA v0.8 — runs for the weekend. Usage:  NPROC=16 bash run_all.sh  [songs|toy|mnist|all]
# Each block writes its own .pkl files and a report_*.txt. Blocks can be run separately and in any order.
set -eo pipefail
export NPROC=${NPROC:-$(nproc)}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1   # one thread per worker process
WHAT=${1:-all}
mkdir -p results logs

# ---------------------------------------------------------------- 1. melodies (Section 5), full loop
if [[ $WHAT == songs || $WHAT == all ]]; then
  for P in 1 0; do                       # 1 = pitch-only context (as in v0.8), 0 = pitch + onset (factored)
    TAG=$([[ $P == 1 ]] && echo pitch || echo factored)
    PITCH=$P OUT=results/songs_results_$TAG.pkl python exp_songs.py 0 29 | tee logs/songs_$TAG.log
    python analyse_songs.py results/songs_results_$TAG.pkl > results/report_songs_$TAG.txt
    python plot_songs.py results/songs_results_$TAG.pkl results/fig_v08_songs_$TAG.pdf
    python plot_songs_controls.py results/songs_results_$TAG.pkl results/fig_songs_controls_$TAG.pdf
  done
fi

# ---------------------------------------------------------------- 2. synthetic world (Section 3) with the full loop
# lam = gain of the returning error; inhibition at the k-th winner; NU_THR = novelty threshold in units of k
if [[ $WHAT == toy || $WHAT == all ]]; then
  for CM in 0 0.5; do                    # depression of excess units (v0.7 used 0.5; the melodies use 0)
  for LAM in 0.05 0.2; do
    for THR in 2 4 8; do
      TAG=c${CM}_l${LAM}_t${THR}
      export ENGDOPA_CMINUS=$CM ENGDOPA_LAM=$LAM ENGDOPA_INH=kth NU_THR=$THR
      if [[ $LAM == 0.2 && $THR == 2 ]]; then export E_THRS=2,4,8; fi     # reviewer control: error-count trigger
      OUT=results/toy_$TAG.pkl python exp_toy.py 20 | tee logs/toy_$TAG.log
      python analyse_toy.py results/toy_$TAG.pkl > results/report_toy_$TAG.txt
      OUT=results/toyfix_$TAG.pkl python exp_toy_fix.py 20 | tee -a logs/toy_$TAG.log
      python analyse_toy_fix.py results/toyfix_$TAG.pkl > results/report_toyfix_$TAG.txt
      unset ENGDOPA_CMINUS ENGDOPA_LAM ENGDOPA_INH NU_THR E_THRS
    done
  done
  done
  # reference: the v0.7 feedforward signal, to check that the numbers of v0.7.2 are reproduced
  OUT=results/toy_feedforward.pkl python exp_toy.py 20 | tee logs/toy_feedforward.log
  python analyse_toy.py results/toy_feedforward.pkl > results/report_toy_feedforward.txt
fi

# ---------------------------------------------------------------- 3. class-incremental MNIST (Section 4) with the full loop
if [[ $WHAT == mnist || $WHAT == all ]]; then
  for CM in 0 0.5; do
  for LAM in 0.05 0.2; do
    for THR in 1 2 4; do
      TAG=c${CM}_l${LAM}_t${THR}
      export ENGDOPA_CMINUS=$CM ENGDOPA_LAM=$LAM ENGDOPA_INH=kth NU_THR=$THR
      if [[ $LAM == 0.2 && $THR == 2 ]]; then export E_THRS=2,5,10; fi    # reviewer control: error-count trigger
      OUT=results/bench_$TAG.pkl python exp_bench.py 1 11 | tee logs/bench_$TAG.log
      OUT=results/bench_$TAG.pkl python exp_bench2.py 1 11 | tee -a logs/bench_$TAG.log
      python analyse_bench2.py results/bench_$TAG.pkl > results/report_bench_$TAG.txt
      OUT=results/nu_mnist_$TAG.pkl python nu_mnist.py > results/report_nu_mnist_$TAG.txt
      unset ENGDOPA_CMINUS ENGDOPA_LAM ENGDOPA_INH NU_THR E_THRS
    done
  done
  done
fi
echo "finished: $(date)"
