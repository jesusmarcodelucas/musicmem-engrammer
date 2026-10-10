"""v0.7.1 controls: ablation on the fixed trigger sequence of the joint condition; causal adaptive surprise trigger."""
import numpy as np, pickle, time, os, sys
from multiprocessing import Pool
from world import make_world, make_stream
import toy, exp_toy
def one(seed, share=8):
    rng = np.random.default_rng(seed * 100 + share); w = make_world(rng, share=share); st = make_stream(rng, w)
    enc, pred, prev = toy.pretrain(rng, st, eta=exp_toy.ETA)
    full = toy.run_test(enc, pred, prev, w, st, trigger='N', nu_thr=exp_toy.THR, boost=exp_toy.BOOST, record=True)
    trig = full['rec']['trig']
    out = {'full': {k: v for k, v in full.items() if k != 'rec'}}
    out['fixed_alloc_only'] = toy.run_test(enc, pred, prev, w, st, trigger='shuf', replay_trig=trig, boost=0.0)
    out['fixed_gain_only'] = toy.run_test(enc, pred, prev, w, st, trigger='shuf', replay_trig=trig, boost=exp_toy.BOOST, alloc=False)
    out['S_online'] = toy.run_test(enc, pred, prev, w, st, trigger='S_online', boost=exp_toy.BOOST)
    return seed, out
if __name__ == '__main__':
    t0 = time.time()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    with Pool(int(os.environ.get('NPROC', 2))) as p: res = dict(p.map(one, range(1, n + 1)))
    pickle.dump(res, open(os.environ.get('OUT', 'toy_fix.pkl'), 'wb')); print('done', round(time.time() - t0), 's')
