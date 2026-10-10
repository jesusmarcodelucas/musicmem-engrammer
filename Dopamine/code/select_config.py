"""Compare the configurations of the grid on the tuning seed only (seed 1), to choose lam/threshold/cminus
without looking at the reported seeds. Usage: python select_config.py"""
import glob, pickle, numpy as np
print('== MNIST, seed 1 only: final accuracy (novelty allocation / every image / shuffled / none)')
for f in sorted(glob.glob('results/bench_c*.pkl')):
    R = pickle.load(open(f, 'rb'))
    if 1 not in R: continue
    r = R[1]; row = []
    for N in [1000, 2000]:
        for sc in ['strict', 'recurring']:
            g = lambda c: r.get((N, sc, c), {}).get('final_acc', np.nan)
            row.append(f'{N}/{sc[:3]}: {g("alloc"):.2f}/{g("always_a"):.2f}/{g("shuf_a"):.2f}/{g("none"):.2f}')
    rp = lambda c: r.get((2000, 'strict', c), {}).get('final_acc', np.nan)
    print(f[8:-4], ' | '.join(row), f'| replay {rp("alloc+replay"):.2f}/{rp("always_a+replay"):.2f}/{rp("shuf_a+replay"):.2f}/{rp("replay"):.2f}')
print('== synthetic world, seed 1 only: collisions / parent stability / familiar above threshold / AUC')
for f in sorted(glob.glob('results/toy_c*.pkl')):
    R = pickle.load(open(f, 'rb')); r = R[1]
    print(f[8:-4], ' '.join(f'{c}={r[c]["collide"]:.2f}/{r[c]["parent_stab"]:.2f}' for c in ['none', 'full', 'shuf', 'oracle', 'always']),
          f'fam>thr={r["fam_thr"]:.2f} AUC={r["auc"][0]:.3f}')
