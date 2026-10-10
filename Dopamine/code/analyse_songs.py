import sys, pickle, numpy as np
fn = sys.argv[1] if len(sys.argv) > 1 else 'songs_results_pitch.pkl'
R = pickle.load(open(fn, 'rb')); S = sorted(R); rng = np.random.default_rng(0)
K0 = R[S[0]]
def ms(key, m):
    x = np.array([R[s][key][m] for s in S], float); return x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
def row(key, metrics):
    return ' '.join(f'{m}={ms(key, m)[0]:.3f}±{ms(key, m)[1]:.3f}' for m in metrics if m in K0[key])
print(fn, 'seeds', S, f'(n={len(S)})')
print('\nA: detection and sparsity (frozen latent layer)')
for key in [k for k in K0 if k[0] == 'A']:
    print(' ', key, row(key, ['auc', 'auc_errp', 'auc_err', 'active_train_med', 'active_novel_med', 'frac_train_sparse', 'retention', 'free_recall']))
for ep in [10, 2]:
    print(f'\nB: dopamine-triggered allocation, {ep} passes per song')
    for key in [k for k in K0 if k[0] == 'B' and k[1] == ep]:
        print(' ', key, row(key, ['retention', 'free_recall', 'acquisition', 'auc', 'frac_train_sparse', 'trig_rate', 'used',
                                  'noisy_trig_rate', 'noisy_new_units', 'retention_after_noise']))
    for lam in [0.05, 0.2]:
        if ('B', ep, lam, 'novelty') not in K0: continue
        for other in ['shuffled', 'none', 'every step', 'frozen', 'novelty+gain', 'error count', 'error-count shuffled']:
            if ('B', ep, lam, other) not in K0: continue
            for met in ['retention', 'free_recall']:
                d = boot([R[s][('B', ep, lam, 'novelty')][met] - R[s][('B', ep, lam, other)][met] for s in S])
                print(f'    novelty - {other} (lam={lam}) {met}: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]')
    if ('B', ep, 0.2, 'error count') in K0:
        d = boot([R[s][('B', ep, 0.2, 'error count')]['retention'] - R[s][('B', ep, 0.2, 'error-count shuffled')]['retention'] for s in S])
        print(f'    error count - its shuffle retention: {d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]')
print('\nC: retention against pulse rate (thresholds on nu and on sum C+)')
for key in [k for k in K0 if k[0] == 'C']:
    print(' ', key, row(key, ['trig_rate', 'retention', 'free_recall', 'used']))
print('\nProbes: fraction flagged by nu>0 / by sum(C+)>0, and mean excess units')
PK = [k for k in K0 if isinstance(K0[k], dict) and 'probe' in K0[k]]
for key in PK:
    print(' ', key)
    for pk in K0[key]['probe']:
        f = lambda m: np.nanmean([R[s][key]['probe'][pk][m] for s in S])
        extra = f"  n={int(np.nanmean([R[s][key]['probe'][pk]['n'] for s in S]))}" if 'n' in K0[key]['probe'][pk] else f"  excess={f('nu'):.2f}"
        print(f'    {str(pk):28s} nu: {f("flag_nu"):.3f}  sumC+: {f("flag_err"):.3f}{extra}')
