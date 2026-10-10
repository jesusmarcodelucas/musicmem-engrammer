"""Batch C: class-incremental MNIST, signed-XOR only (reduced configuration)."""
import numpy as np, pickle, sys, time, os
from multiprocessing import Pool
from bench import run
d = np.load(os.environ.get('MNIST14', 'mnist14.npz')); Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
B = 3.0
def one(seed):
    out = {}
    for N in [1000, 2000]:
        for sc, po in [('strict', 0.0), ('recurring', 0.5)]:
            kw = dict(N=N, p_old=po)
            out[(N, sc, 'none')] = run(Xtr, ytr, Xte, yte, seed, 'none', **kw)
            out[(N, sc, 'always')] = run(Xtr, ytr, Xte, yte, seed, 'always', boost=B, **kw)
            out[(N, sc, 'alloc')] = run(Xtr, ytr, Xte, yte, seed, 'N', **kw)
            full = run(Xtr, ytr, Xte, yte, seed, 'N', boost=B, record=True, **kw); full.pop('trig_arr', None)
            perm = np.random.default_rng(seed + 7).permutation(full['trig'])
            out[(N, sc, 'full')] = {k: v for k, v in full.items() if k != 'trig'}
            out[(N, sc, 'shuf')] = run(Xtr, ytr, Xte, yte, seed, 'shuf', replay_trig=perm, boost=B, **kw)
    out[(2000, 'strict', 'err')] = run(Xtr, ytr, Xte, yte, seed, 'err', boost=B, N=2000)
    for name, kw in [('replay', dict(trigger='none')), ('always+replay', dict(trigger='always', boost=B)),
                     ('full+replay', dict(trigger='N', boost=B, record=True))]:
        r = run(Xtr, ytr, Xte, yte, seed, replay=500, N=2000, **kw)
        if name == 'full+replay':
            perm = np.random.default_rng(seed + 8).permutation(r.pop('trig'))
            out[(2000, 'strict', 'shuf+replay')] = run(Xtr, ytr, Xte, yte, seed, 'shuf', replay_trig=perm, boost=B, replay=500, N=2000)
        out[(2000, 'strict', name)] = r
    out[(2000, 'iid', 'none')] = run(Xtr, ytr, Xte, yte, seed, 'none', iid=True, N=2000)
    out[(2000, 'iid', 'full')] = run(Xtr, ytr, Xte, yte, seed, 'N', boost=B, iid=True, N=2000)
    return seed, out
if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    fn = os.environ.get('OUT', 'bench_results.pkl')
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        for seed, out in p.imap_unordered(one, range(a, b + 1)):
            res[seed] = out; pickle.dump(res, open(fn, 'wb'))
    print('done', sorted(res), round(time.time() - t0), 's')
