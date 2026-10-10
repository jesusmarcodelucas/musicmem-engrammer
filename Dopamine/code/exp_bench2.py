"""Batch C2: allocation-only controls (no plasticity boost) for the MNIST benchmark."""
import numpy as np, pickle, sys, time, os
from multiprocessing import Pool
from bench import run
d = np.load(os.environ.get('MNIST14', 'mnist14.npz')); Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
def one(seed):
    out = {}
    for N in [1000, 2000]:
        for sc, po in [('strict', 0.0), ('recurring', 0.5)]:
            kw = dict(N=N, p_old=po)
            a = run(Xtr, ytr, Xte, yte, seed, 'N', record=True, **kw)
            perm = np.random.default_rng(seed + 7).permutation(a.pop('trig'))
            out[(N, sc, 'alloc')] = a
            out[(N, sc, 'always_a')] = run(Xtr, ytr, Xte, yte, seed, 'always', **kw)
            out[(N, sc, 'shuf_a')] = run(Xtr, ytr, Xte, yte, seed, 'shuf', replay_trig=perm, **kw)
            for et in [float(v) for v in os.environ.get('E_THRS', '').split(',') if v]:   # reviewer control: sum(C+) trigger
                out[(N, sc, f'E{et:g}_a')] = run(Xtr, ytr, Xte, yte, seed, 'E', e_thr=et, **kw)
    out[(2000, 'strict', 'err_a')] = run(Xtr, ytr, Xte, yte, seed, 'err', N=2000)
    r = run(Xtr, ytr, Xte, yte, seed, 'N', replay=500, N=2000, record=True)
    perm = np.random.default_rng(seed + 8).permutation(r.pop('trig'))
    out[(2000, 'strict', 'alloc+replay')] = r
    out[(2000, 'strict', 'always_a+replay')] = run(Xtr, ytr, Xte, yte, seed, 'always', replay=500, N=2000)
    out[(2000, 'strict', 'shuf_a+replay')] = run(Xtr, ytr, Xte, yte, seed, 'shuf', replay_trig=perm, replay=500, N=2000)
    out[(2000, 'iid', 'alloc')] = run(Xtr, ytr, Xte, yte, seed, 'N', iid=True, N=2000)
    return seed, out
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time(); fn = os.environ.get('OUT', 'bench_results.pkl')
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        for seed, out in p.imap_unordered(one, range(a, b + 1)):
            res.setdefault(seed, {}).update(out); pickle.dump(res, open(fn, 'wb'))
    print('done', round(time.time() - t0), 's')
