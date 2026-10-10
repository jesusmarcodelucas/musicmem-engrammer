import numpy as np, pickle, sys
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'bench_results.pkl', 'rb')); S = [s for s in sorted(R) if s != 1]
rng = np.random.default_rng(0)
def boot(x, n=10000):
    x = np.asarray(x, float); b = x[rng.integers(0, len(x), (n, len(x)))].mean(1); return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)
f = lambda t: f'{t[0]:+.3f} [{t[1]:+.3f},{t[2]:+.3f}]'
summ = {}
for N in [1000, 2000]:
    for sc in ['strict', 'recurring']:
        print(f'== N={N} {sc}')
        EC = sorted({k[2] for s in S for k in R[s] if len(k) == 3 and str(k[2]).startswith('E')})
        for c in ['none', 'always_a', 'shuf_a', 'alloc', 'full'] + EC:
            fa = boot([R[s][(N, sc, c)]['final_acc'] for s in S]); fo = boot([R[s][(N, sc, c)]['forgetting'] for s in S])
            tr = np.mean([R[s][(N, sc, c)]['trig_rate'] for s in S])
            summ[(N, sc, c)] = (fa, fo, tr); print(f'  {c:9s} final {fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}] forg {fo[0]:.3f} trig {tr:.2f}')
        for a, b in [('alloc', 'none'), ('alloc', 'always_a'), ('alloc', 'shuf_a'), ('alloc', 'full')] + [('alloc', e) for e in EC]:
            d = boot([R[s][(N, sc, a)]['final_acc'] - R[s][(N, sc, b)]['final_acc'] for s in S]); summ[(N, sc, a, b)] = d
            print(f'    {a}-{b}: {f(d)}')
print('== N=2000 strict, replay and error trigger')
for c in ['err_a', 'replay', 'always_a+replay', 'shuf_a+replay', 'alloc+replay']:
    fa = boot([R[s][(2000, 'strict', c)]['final_acc'] for s in S]); fo = boot([R[s][(2000, 'strict', c)]['forgetting'] for s in S])
    summ[(2000, 'strict', c)] = (fa, fo); print(f'  {c:16s} final {fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}] forg {fo[0]:.3f}')
for a, b in [('alloc+replay', 'replay'), ('alloc+replay', 'always_a+replay'), ('alloc+replay', 'shuf_a+replay'), ('alloc', 'err_a')]:
    d = boot([R[s][(2000, 'strict', a)]['final_acc'] - R[s][(2000, 'strict', b)]['final_acc'] for s in S]); summ[(2000, 'strict', a, b)] = d
    print(f'    {a}-{b}: {f(d)}')
for c in ['none', 'alloc']:
    fa = boot([R[s][(2000, 'iid', c)]['final_acc'] for s in S]); summ[(2000, 'iid', c)] = (fa,); print('iid', c, f'{fa[0]:.3f} [{fa[1]:.3f},{fa[2]:.3f}]')
pickle.dump(summ, open((sys.argv[1] if len(sys.argv) > 1 else 'bench_results.pkl').replace('.pkl', '_summary2.pkl'), 'wb'))
