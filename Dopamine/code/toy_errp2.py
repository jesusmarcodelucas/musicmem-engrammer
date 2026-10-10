import os, sys, numpy as np, pickle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RES = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
import exp_toy, toy
from engdopa import Encoder
from world import make_world, make_stream
LOG = []; _r = Encoder.respond
def resp(self, x):
    o = _r(self, x); LOG.append((self.last_errp, o[3])); return o
R = {}
for seed in [int(a) for a in sys.argv[1:]]:
    rng = np.random.default_rng(seed * 100 + 8); w = make_world(rng, share=8); st = make_stream(rng, w)
    enc, pred, prev = toy.pretrain(rng, st, eta=exp_toy.ETA); T0 = st['T0']
    Encoder.respond = resp; LOG.clear()
    toy.run_test(enc, pred, prev, w, st, trigger='none'); Encoder.respond = _r
    E = np.array([l[0] for l in LOG]); NU = np.array([l[1] for l in LOG]) * toy.K
    it = st['items'][T0:]; seen = set(range(w['P_old'])); first = np.zeros(len(it), bool)
    for i, x in enumerate(it):
        first[i] = x < 0 or x not in seen; seen.add(int(x))
    fam = (it >= 0) & (it < w['P_old']); newf = (it >= w['P_old']) & first; one = it < 0
    R[seed] = dict(E_fam=E[fam], E_new=E[newf], E_one=E[one], N_fam=NU[fam], N_new=NU[newf], auc_E=exp_toy.auc(E[first], E[fam]), auc_N=exp_toy.auc(NU[first], NU[fam]))
pickle.dump(R, open(os.path.join(RES, 'extra', 'toy_errp_%s.pkl' % sys.argv[1]), 'wb'))
