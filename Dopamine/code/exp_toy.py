"""Batch A+B: novelty signal and allocation controls in the toy world (signed-XOR network)."""
import numpy as np, pickle, time, sys, os
from multiprocessing import Pool
from world import make_world, make_stream, D, S_BITS
import toy
from engdopa import Encoder
ETA, GAMMA, BOOST = 0.02, 0.85, 3.0
THR = float(os.environ.get('NU_THR', 1.0))   # novelty threshold (in units of k)
_orig = Encoder.__init__
def _init(self, *a, **kw):
    kw.update(eta=ETA, gamma=GAMMA); _orig(self, *a, **kw)
Encoder.__init__ = _init

def auc(pos, neg):
    allv = np.r_[pos, neg]; order = allv.argsort(kind='mergesort'); r = np.empty(len(allv)); r[order] = np.arange(1, len(allv) + 1)
    u, inv = np.unique(allv, return_inverse=True)
    for i in range(len(u)):
        m = inv == i; r[m] = r[m].mean()
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

def one(seed, share=8):
    rng = np.random.default_rng(seed * 100 + share); w = make_world(rng, share=share); st = make_stream(rng, w)
    enc, pred, prev = toy.pretrain(rng, st, eta=ETA); T0 = st['T0']
    out = {}
    base = toy.run_test(enc, pred, prev, w, st, trigger='none', record=True)
    out['none'] = {k: v for k, v in base.items() if k != 'rec'}
    # signals by event category
    lab = st['lab'][T0:]; it = st['items'][T0:]; Nn = base['rec']['N'] * toy.K; Ss = base['rec']['S']
    seen = set(range(w['P_old'])); first = np.zeros(len(it), bool)
    for i, x in enumerate(it):
        first[i] = x < 0 or x not in seen; seen.add(int(x))
    cat = np.full(len(it), 0)
    cat[lab == 1] = 1; cat[lab == 2] = 2; cat[lab == 3] = 3
    cat[(it >= w['P_old']) & first] = 4; cat[(it >= w['P_old']) & ~first] = 5; cat[it < 0] = 6
    cat[(lab == 5) & (it < w['P_old']) & (it >= 0)] = 0
    out['sig'] = {c: (Nn[cat == c].mean(), Ss[cat == c].mean()) for c in range(7)}
    novel = first; fam = (it >= 0) & (it < w['P_old'])
    out['auc'] = (auc(Nn[novel], Nn[fam]), auc(Ss[novel], Ss[fam]))
    out['fam_any'] = np.mean(Nn[fam] > 0); out['fam_thr'] = np.mean(Nn[fam] > THR * toy.K)
    out['second_visit'] = Nn[(it >= w['P_old']) & ~first].mean()
    # assimilation probes (no learning)
    prng = np.random.default_rng(seed + 5000); dens = {}
    for sh in range(0, 17, 2):
        dd = []
        for _ in range(20):
            p = prng.integers(w['P_old']); on = np.where(w['pats'][p])[0]; off = np.where(~w['pats'][p])[0]
            x = np.zeros(D); x[prng.choice(on, sh, replace=False)] = 1; x[prng.choice(off, S_BITS - sh, replace=False)] = 1
            dd.append(enc.respond(x)[1].sum())
        dens[sh] = np.array(dd)
    out['dens'] = dens
    if share is None: return out
    # allocation conditions
    full = toy.run_test(enc, pred, prev, w, st, trigger='N', nu_thr=THR, boost=BOOST, record=True)
    out['full'] = {k: v for k, v in full.items() if k != 'rec'}
    n = int(full['n_trig'])
    out['alloc_only'] = toy.run_test(enc, pred, prev, w, st, trigger='N', nu_thr=THR, boost=0.0)
    out['plast_only'] = toy.run_test(enc, pred, prev, w, st, trigger='N', nu_thr=THR, boost=BOOST, alloc=False)
    perm = np.random.default_rng(seed + 7).permutation(full['rec']['trig'])
    out['shuf'] = toy.run_test(enc, pred, prev, w, st, trigger='shuf', replay_trig=perm, boost=BOOST)
    out['oracle'] = toy.run_test(enc, pred, prev, w, st, trigger='oracle', boost=BOOST)
    topS = np.zeros(len(Ss), bool); topS[np.argsort(-Ss, kind='stable')[:n]] = True   # the n most surprising steps
    out['S_matched'] = toy.run_test(enc, pred, prev, w, st, trigger='shuf', replay_trig=topS, boost=BOOST)
    out['always'] = toy.run_test(enc, pred, prev, w, st, trigger='always', boost=BOOST)
    # reviewer control: the reconstruction-error count sum(C+) as the trigger, for several thresholds (bits)
    for et in [float(v) for v in os.environ.get('E_THRS', '').split(',') if v]:
        out[f'E{et:g}'] = toy.run_test(enc, pred, prev, w, st, trigger='E', e_thr=et, boost=BOOST)
    return out

def job(seed):
    return seed, one(seed)

if __name__ == '__main__':
    seeds = range(1, int(sys.argv[1]) + 1) if len(sys.argv) > 1 else range(1, 21)
    t0 = time.time()
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        res = dict(p.map(job, seeds))
    pickle.dump(res, open(os.environ.get('OUT', 'toy_results.pkl'), 'wb')); print('done', round(time.time() - t0), 's')
