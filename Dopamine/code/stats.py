import pickle, numpy as np
rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
import os
RES = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
P = lambda sub, f: os.path.join(RES, sub, f) if os.path.exists(os.path.join(RES, sub, f)) else os.path.join(RES, f)   # results/<sub>/ (repository) or flat results/ (run_all.sh)
U = RES
# ---------------- toy
T = pickle.load(open(P('toy', f'toy_c0_l0.2_t8.pkl'), 'rb')); TS = sorted(T)
TE = pickle.load(open(P('toy', f'toy_c0_l0.2_t2.pkl'), 'rb'))
TF = pickle.load(open(P('toy', f'toyfix_c0_l0.2_t8.pkl'), 'rb'))
for s in TS:
    for e in ['E2', 'E4', 'E8']: T[s][e] = TE[s][e]
    for c in ['fixed_alloc_only', 'fixed_gain_only', 'S_online']: T[s][c] = TF[s][c]
print('== TOY c0 l0.2 t8, n =', len(TS))
CATS = ['familiar', 'stochastic', 'reset', 'rewired', 'new 1st', 'new later', 'one-off']
print(' signal by category (excess units, surprise):', {CATS[c]: (round(np.mean([T[s]['sig'][c][0] for s in TS]), 2), round(np.mean([T[s]['sig'][c][1] for s in TS]), 1)) for c in range(7)})
print(' AUC nu', f(boot([T[s]['auc'][0] for s in TS])), 'AUC S', f(boot([T[s]['auc'][1] for s in TS])))
print(' fam_any', round(np.mean([T[s]['fam_any'] for s in TS]), 3), 'fam_thr', round(np.mean([T[s]['fam_thr'] for s in TS]), 3), 'second visit', round(np.mean([T[s]['second_visit'] for s in TS]), 2))
for sh in range(0, 17, 2):
    d = np.concatenate([T[s]['dens'][sh] for s in TS]); print(f'  probe share {sh*5}%: median active {np.median(d):.0f}, identified {np.mean(d <= 5):.2f}')
K = ['collide', 'parent_stab', 'old_stab', 'pred_new', 'pred_old', 'pred_rew', 'n_trig']
for c in ['none', 'plast_only', 'alloc_only', 'full', 'shuf', 'S_matched', 'S_online', 'oracle', 'always', 'fixed_alloc_only', 'fixed_gain_only', 'E2', 'E4', 'E8']:
    print(f'  {c:17s}', ' '.join(f'{k}={np.mean([T[s][c][k] for s in TS]):.3f}' for k in K))
for b in ['none', 'shuf', 'S_matched', 'oracle', 'always', 'fixed_alloc_only', 'fixed_gain_only', 'E8', 'S_online']:
    print(f'  full-{b}:', ' '.join(f'{k}={f(boot([T[s]["full"][k]-T[s][b][k] for s in TS]))}' for k in ['collide', 'parent_stab', 'pred_new', 'pred_old']))
# ---------------- MNIST
M = pickle.load(open(P('mnist', f'bench_c0_l0.2_t1.pkl'), 'rb')); MS = [s for s in sorted(M) if s != 1]
ME = pickle.load(open(P('mnist', f'bench_c0_l0.2_t2.pkl'), 'rb'))
print('== MNIST c0 l0.2 t1, seeds', MS)
C = ['none', 'err_a', 'always_a', 'shuf_a', 'alloc', 'full']
for N in [1000, 2000]:
    for sc in ['strict', 'recurring']:
        row = []
        for c in C:
            if (N, sc, c) in M[MS[0]]:
                row.append(f"{c}={np.mean([M[s][(N, sc, c)]['final_acc'] for s in MS]):.3f}(trig {np.mean([M[s][(N, sc, c)]['trig_rate'] for s in MS]):.2f}, forg {np.mean([M[s][(N, sc, c)]['forgetting'] for s in MS]):.2f})")
        print(f' {N} {sc}:', ' '.join(row))
        for b in ['none', 'always_a', 'shuf_a']:
            print(f'    alloc-{b}: {f(boot([M[s][(N,sc,"alloc")]["final_acc"]-M[s][(N,sc,b)]["final_acc"] for s in MS]))}')
        for e in ['E2_a', 'E5_a', 'E10_a']:
            print(f'    t2 file {e}: {np.mean([ME[s][(N,sc,e)]["final_acc"] for s in MS]):.3f} trig {np.mean([ME[s][(N,sc,e)]["trig_rate"] for s in MS]):.2f}', end='')
        print(f'   | t2 novelty {np.mean([ME[s][(N,sc,"alloc")]["final_acc"] for s in MS]):.3f} trig {np.mean([ME[s][(N,sc,"alloc")]["trig_rate"] for s in MS]):.2f}')
for c in ['err_a', 'replay', 'always_a+replay', 'shuf_a+replay', 'alloc+replay']:
    print(f'  2000 strict {c}: {np.mean([M[s][(2000,"strict",c)]["final_acc"] for s in MS]):.3f}')
for b in ['replay', 'always_a+replay', 'shuf_a+replay']:
    print(f'    alloc+replay-{b}: {f(boot([M[s][(2000,"strict","alloc+replay")]["final_acc"]-M[s][(2000,"strict",b)]["final_acc"] for s in MS]))}')
print('  iid none/alloc/full', [round(np.mean([M[s][(2000, 'iid', c)]['final_acc'] for s in MS]), 3) for c in ['none', 'alloc', 'full']])
print('  used units', {c: int(np.mean([M[s][(2000, 'strict', c)]['used'] for s in MS])) for c in ['none', 'alloc', 'always_a']})
# nu_mnist (threshold nu > 1, in units of k)
NM = pickle.load(open(P('mnist', f'nu_mnist_c0_l0.2_t1.pkl'), 'rb')); thr = 1.0
for name in ['rec1000', 'strict2000_replay']:
    print('==', name)
    for b in range(5):
        nn = np.concatenate([NM[s][name]['nu'][(NM[s][name]['blk'] == b) & NM[s][name]['new']] for s in NM])
        no = np.concatenate([NM[s][name]['nu'][(NM[s][name]['blk'] == b) & ~NM[s][name]['new']] for s in NM])
        print(f'  block {b+1}: new-class above thr {np.mean(nn > thr):.2f}' + (f' | old-class {np.mean(no > thr):.2f}' if len(no) else ''))
    rp = np.concatenate([NM[s][name]['nu_rep'] for s in NM])
    if len(rp): print('  replayed images above thr', round(float(np.mean(rp > thr)), 2))
