import numpy as np, pickle, sys
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'toy_results.pkl', 'rb')); S = sorted(R); rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
CATS = ['familiar', 'stochastic', 'reset', 'rewired', 'new 1st', 'new later', 'one-off']
for c in range(7):
    print(f'{CATS[c]:10s} N={np.mean([R[s]["sig"][c][0] for s in S]):.2f}  S={np.mean([R[s]["sig"][c][1] for s in S]):.1f}')
for k in ['fam_any', 'fam_thr', 'second_visit']:
    print(k, f(boot([R[s][k] for s in S])))
print('AUC N', f(boot([R[s]['auc'][0] for s in S])), 'AUC S', f(boot([R[s]['auc'][1] for s in S])))
for sh in range(0, 17, 2):
    d = np.concatenate([R[s]['dens'][sh] for s in S]); print('share', sh, 'median', np.median(d), 'ident', round(np.mean(d <= 5), 2))
conds = ['none', 'plast_only', 'alloc_only', 'full', 'shuf', 'oracle', 'S_matched', 'always'] + [c for c in R[S[0]] if c.startswith('E')]
keys = ['collide', 'parent_stab', 'old_stab', 'acq', 'pred_new', 'pred_old', 'pred_rew', 'totV', 'n_trig']
summ = {}
for c in conds:
    row = {k: np.mean([R[s][c][k] for s in S]) for k in keys}; summ[c] = row
    print(f'{c:11s}', ' '.join(f'{k}={row[k]:.3f}' if k not in ('totV', 'n_trig') else f'{k}={row[k]:.0f}' for k in keys))
for a, b in [(('full', c)) for c in R[S[0]] if c.startswith('E')] + [('full', 'none'), ('plast_only', 'none'), ('alloc_only', 'none'), ('full', 'shuf'), ('full', 'oracle'), ('full', 'S_matched'), ('full', 'always'), ('full', 'plast_only'), ('full', 'alloc_only')]:
    d = {k: boot([R[s][a][k] - R[s][b][k] for s in S]) for k in ['collide', 'parent_stab', 'old_stab', 'pred_new', 'pred_old', 'pred_rew']}
    summ[(a, b)] = d
    print(f'{a}-{b}:', ' '.join(f'{k}={f(v)}' for k, v in d.items()))
pickle.dump(summ, open((sys.argv[1] if len(sys.argv) > 1 else 'toy_results.pkl').replace('.pkl', '_summary.pkl'), 'wb'))
