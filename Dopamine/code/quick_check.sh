#!/usr/bin/env bash
# 2-minute smoke test before launching run_all.sh
set -eo pipefail
export OMP_NUM_THREADS=1 NPROC=2
PITCH=1 OUT=/tmp/qc_songs.pkl python exp_songs.py 0 1 && python analyse_songs.py /tmp/qc_songs.pkl > /tmp/qc_songs.txt && grep -A3 'passes per song' /tmp/qc_songs.txt || true
ENGDOPA_CMINUS=0 ENGDOPA_LAM=0.2 ENGDOPA_INH=kth NU_THR=2 python -c "
import numpy as np; from bench import run
d=np.load('mnist14.npz'); r=run(d['Xtr'],d['ytr'],d['Xte'],d['yte'],2,'N',N=1000,n_per_task=200,n_test=500)
print('mnist smoke test ok, final acc', round(r['final_acc'],3), 'trig rate', round(r['trig_rate'],3))
r=run(d['Xtr'],d['ytr'],d['Xte'],d['yte'],2,'E',e_thr=5,N=1000,n_per_task=200,n_test=500)
print('mnist error-count trigger ok, final acc', round(r['final_acc'],3), 'trig rate', round(r['trig_rate'],3))"
E_THRS=4 ENGDOPA_CMINUS=0 ENGDOPA_LAM=0.2 ENGDOPA_INH=kth NU_THR=4 OUT=/tmp/qc_toy.pkl python exp_toy.py 1
python -c "import pickle; r=pickle.load(open('/tmp/qc_toy.pkl','rb'))[1]; print('toy ok: collisions none/novelty/error-count', r['none']['collide'], r['full']['collide'], r['E4']['collide'])"
rm -f /tmp/qc_songs.pkl /tmp/qc_songs.txt /tmp/qc_toy.pkl; echo OK
