import sys, pickle, numpy as np
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'toy_fix.pkl', 'rb')); S = sorted(R); rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
keys = ['collide', 'parent_stab', 'old_stab', 'pred_new', 'pred_old', 'n_trig']
for c in R[S[0]]:
    print(f'{c:17s}', ' '.join(f'{k}={np.mean([R[s][c][k] for s in S]):.3f}' for k in keys))
for b in ['fixed_alloc_only', 'fixed_gain_only', 'S_online']:
    d = boot([R[s]['full']['collide'] - R[s][b]['collide'] for s in S]); print(f'full - {b} collisions: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]')
