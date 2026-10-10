import os, sys, numpy as np, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
import exp_toy, toy
from engdopa import Encoder
from world import make_world, make_stream
LOG = []; _s = Encoder.select
def sel(self, a, D, mode='homeo'):
    c = _s(self, a, D, mode); LOG.append((D > 0, int((self.count[c] == 0).sum()))); return c
out = {}
for seed in [int(a) for a in sys.argv[1:]]:
    rng = np.random.default_rng(seed * 100 + 8); w = make_world(rng, share=8); st = make_stream(rng, w)
    enc, pred, prev = toy.pretrain(rng, st, eta=exp_toy.ETA); T0 = st['T0']
    used0 = int((enc.count > 0).sum())
    Encoder.select = sel; LOG.clear()
    toy.run_test(enc, pred, prev, w, st, trigger='N', nu_thr=8.0, boost=3.0); Encoder.select = _s
    trig = np.array([l[0] for l in LOG]); fresh = np.array([l[1] for l in LOG])
    it = st['items'][T0:]; seen = set(range(w['P_old'])); first = np.zeros(len(it), bool)
    for i, x in enumerate(it):
        first[i] = x < 0 or x not in seen; seen.add(int(x))
    one = it < 0; newf = (it >= w['P_old']) & first; newl = (it >= w['P_old']) & ~first; fam = (it >= 0) & (it < w['P_old'])
    out[seed] = dict(n_one=one.sum(), trig_one=trig[one].mean(), fresh_one=fresh[one].sum(), fresh_newf=fresh[newf].sum(),
                     fresh_newl=fresh[newl].sum(), fresh_fam=fresh[fam].sum(), fresh_total=fresh.sum(), used0=used0, n_trig=trig.sum(),
                     trig_by=dict(one=trig[one].sum(), newf=trig[newf].sum(), newl=trig[newl].sum(), fam=trig[fam].sum()))
pickle.dump(out, open(os.path.join(RES, 'extra', 'toy_noise_%s.pkl' % sys.argv[1]), 'wb'))
