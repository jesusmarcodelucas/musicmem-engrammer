import numpy as np, pickle, sys
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'bench_results.pkl', 'rb')); S = [s for s in sorted(R) if s != 1]   # seed 1 used for tuning
rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
print('seeds', S)
summ = {}
for N in [1000, 2000]:
    for sc in ['strict', 'recurring']:
        print(f'== N={N} {sc}')
        for c in ['none', 'always', 'shuf', 'alloc', 'full']:
            fa = boot([R[s][(N, sc, c)]['final_acc'] for s in S]); fo = np.mean([R[s][(N, sc, c)]['forgetting'] for s in S])
            tr = np.mean([R[s][(N, sc, c)]['trig_rate'] for s in S]); us = np.mean([R[s][(N, sc, c)]['used'] for s in S])
            summ[(N, sc, c)] = (fa, fo, tr, us)
            print(f'  {c:7s} final {fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}] forg {fo:.3f} trig {tr:.2f} used {us:.0f}')
        for a, b in [('full', 'none'), ('full', 'always'), ('full', 'shuf'), ('full', 'alloc')]:
            d = boot([R[s][(N, sc, a)]['final_acc'] - R[s][(N, sc, b)]['final_acc'] for s in S]); summ[(N, sc, a, b)] = d
            print(f'    {a}-{b}: {f(d)}')
print('== N=2000 strict extras')
for c in ['err', 'replay', 'always+replay', 'shuf+replay', 'full+replay']:
    fa = boot([R[s][(2000, 'strict', c)]['final_acc'] for s in S]); fo = np.mean([R[s][(2000, 'strict', c)]['forgetting'] for s in S])
    summ[(2000, 'strict', c)] = (fa, fo); print(f'  {c:14s} final {fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}] forg {fo:.3f}')
for a, b in [('full+replay', 'replay'), ('full+replay', 'always+replay'), ('full+replay', 'shuf+replay')]:
    d = boot([R[s][(2000, 'strict', a)]['final_acc'] - R[s][(2000, 'strict', b)]['final_acc'] for s in S]); summ[(2000, 'strict', a, b)] = d
    print(f'    {a}-{b}: {f(d)}')
for c in ['none', 'full']:
    fa = boot([R[s][(2000, 'iid', c)]['final_acc'] for s in S]); summ[(2000, 'iid', c)] = (fa,); print('iid', c, f'{fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}]')
pickle.dump(summ, open((sys.argv[1] if len(sys.argv) > 1 else 'bench_results.pkl').replace('.pkl', '_summary.pkl'), 'wb'))
