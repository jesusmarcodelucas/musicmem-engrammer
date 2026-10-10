import numpy as np, pickle
import os
RES = os.environ.get('RESULTS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results'))
FIG = os.environ.get('FIGDIR', '.'); os.makedirs(FIG, exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, AQUA, ORANGE, G1, G2 = '#2a78d6', '#1baf7a', '#eb6834', '#6f6e69', '#3a3936'
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6})
def tidy(ax, axis='y'):
    ax.grid(color=GRID, lw=0.6, axis=axis); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
# ---------------- Figure: toy world
P = lambda sub, f: os.path.join(RES, sub, f) if os.path.exists(os.path.join(RES, sub, f)) else os.path.join(RES, f)   # results/<sub>/ (repository) or flat results/ (run_all.sh)
U = RES; T = pickle.load(open(P('toy', f'toy_c0_l0.2_t8.pkl'), 'rb')); TS = sorted(T)
TE = pickle.load(open(P('toy', f'toy_c0_l0.2_t2.pkl'), 'rb'))
for s in TS: T[s]['E8'] = TE[s]['E8']
fig, axs = plt.subplots(2, 3, figsize=(7.2, 5.2))
CATS = ['familiar (noisy)', 'stochastic branch', 'reset', 'rewired transition', 'new item 1st visit', 'new item later visits', 'one-off input']
y = np.arange(7)[::-1]
for ax, idx, col, ttl, xl, lab in [(axs[0, 0], 0, BLUE, 'a  Novelty: excess active units', 'active units above the set-point k', True),
                                    (axs[0, 1], 1, ORANGE, 'b  Transition surprise', 'surprise of the transition (nats)', False)]:
    v = np.array([[T[s]['sig'][c][idx] for c in range(7)] for s in TS]); m = v.mean(0); e = v.std(0, ddof=1) / np.sqrt(len(TS))
    ax.barh(y, m, xerr=e, color=col, height=0.65, error_kw=dict(ecolor=INK2, lw=0.8))
    for yi, vi in zip(y, m):
        ax.text(vi + m.max() * 0.02, yi, f'{vi:.1f}' if vi >= 1 else f'{vi:.2f}', va='center', fontsize=6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(CATS if lab else [], fontsize=6.3); ax.set_xlim(0, m.max() * 1.22)
    ax.set_xlabel(xl, fontsize=6.8); ax.set_title(ttl, loc='left'); tidy(ax, 'x')
ax = axs[0, 2]; shs = list(range(0, 17, 2)); pct = np.array(shs) / 20 * 100
D = {h: np.concatenate([T[s]['dens'][h] for s in TS]) for h in shs}
ax.fill_between(pct, [np.percentile(D[h], 25) for h in shs], [np.percentile(D[h], 75) for h in shs], color=BLUE, alpha=0.18, lw=0)
ax.plot(pct, [np.median(D[h]) for h in shs], 'o-', color=BLUE, ms=4, mec='white', mew=0.8)
ax.axhline(5, color=MUTED, lw=0.8, ls=':'); ax.text(2, 5.6, 'set-point k = 5', fontsize=6, color=INK2)
ax.set_yscale('log'); ax.set_ylim(3.5, 250); ax.set_yticks([5, 10, 20, 50, 100, 200]); ax.set_yticklabels(['5', '10', '20', '50', '100', '200'])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlabel('shared input bits (%)'); ax.set_ylabel('active latent units (median, IQR)')
ax.set_title('c  Assimilation boundary', loc='left'); tidy(ax)
CONDS = [('none', 'no modulation', G1), ('plast_only', 'plasticity gain only', MUTED), ('alloc_only', 'allocation only', MUTED),
         ('full', 'novelty: allocation + gain', BLUE), ('shuf', 'same, shuffled times', G2), ('S_matched', 'same, surprise times', ORANGE),
         ('E8', r'error count $\Sigma C^+>8$', ORANGE), ('oracle', 'same, oracle times', AQUA), ('always', 'same, every step', G2)]
yy = np.arange(len(CONDS))[::-1]
for ax, key, ttl, xl, lab in [(axs[1, 0], 'collide', 'd  Code collisions', 'new items colliding with parent', True),
                              (axs[1, 1], 'parent_stab', 'e  Familiar engram stability', 'overlap with pre-test code', False),
                              (axs[1, 2], 'pred_new', 'f  New-transition prediction', 'error (into/out of new items)', False)]:
    for yi, (c, l, col) in zip(yy, CONDS):
        v = np.array([T[s][c][key] for s in TS])
        ax.errorbar(v.mean(), yi, xerr=v.std(ddof=1) / np.sqrt(len(v)), fmt='D' if c == 'E8' else 'o', color=col, ms=4.5, mec='white', mew=0.7, elinewidth=0.9, capsize=2)
    ax.set_yticks(yy); ax.set_yticklabels([l for _, l, _ in CONDS] if lab else [], fontsize=6.3)
    ax.set_xlabel(xl, fontsize=6.8); ax.set_title(ttl, loc='left'); tidy(ax, 'x')
fig.tight_layout(); fig.savefig(f'{FIG}/fig_v081_toy.pdf'); fig.savefig(f'{FIG}/fig_v081_toy.png', dpi=150)
# ---------------- Figure: MNIST
R = pickle.load(open(P('mnist', f'bench_c0_l0.2_t1.pkl'), 'rb')); S = [s for s in sorted(R) if s != 1]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
CUR = [('none', 'no modulation', G1, '-', 'o'), ('always_a', 'allocation, every sample', G2, '--', 's'),
       ('shuf_a', 'allocation, shuffled times', MUTED, ':', 'D'), ('alloc', 'allocation by novelty', BLUE, '-', 'o')]
for ax, (N, sc), ttl in [(axs[0], (2000, 'strict'), 'a  Strict stream (N = 2000)'), (axs[1], (1000, 'recurring'), 'b  Old classes recur (N = 1000)')]:
    for c, l, col, ls, mk in CUR:
        v = np.array([R[s][(N, sc, c)]['curve'] for s in S])
        ax.errorbar(np.arange(1, 6), v.mean(0), yerr=v.std(0, ddof=1) / np.sqrt(len(S)), color=col, ls=ls, marker=mk, ms=3.5,
                    mec='white', mew=0.6, elinewidth=0.8, capsize=2, label=l)
    ax.set_ylim(0, 1.02); ax.set_xticks(range(1, 6)); ax.set_xlabel('task (pair of digit classes)')
    ax.set_ylabel('accuracy on classes seen so far'); ax.set_title(ttl, loc='left'); tidy(ax)
ax = axs[2]
RP = [('replay', 'replay only', ORANGE), ('always_a+replay', 'every sample + replay', G2), ('shuf_a+replay', 'shuffled + replay', MUTED), ('alloc+replay', 'novelty + replay', BLUE)]
yy = np.arange(len(RP))[::-1]
for yi, (c, l, col) in zip(yy, RP):
    v = np.array([R[s][(2000, 'strict', c)]['final_acc'] for s in S])
    ax.errorbar(v.mean(), yi, xerr=v.std(ddof=1) / np.sqrt(len(v)), fmt='o', color=col, ms=5, mec='white', mew=0.7, elinewidth=0.9, capsize=2)
io = np.mean([R[s][(2000, 'iid', 'alloc')]['final_acc'] for s in S]); i0 = np.mean([R[s][(2000, 'iid', 'none')]['final_acc'] for s in S])
ax.axvline(io, color=BLUE, lw=0.8, ls='-.'); ax.axvline(i0, color=G1, lw=0.8, ls='-.')
ax.text(io - 0.004, len(RP) - 0.55, 'unordered +\nallocation', fontsize=5.6, color=BLUE, ha='right', va='top')
ax.text(i0 + 0.004, len(RP) - 0.55, 'unordered', fontsize=5.6, color=G1, ha='left', va='top')
ax.set_yticks(yy); ax.set_yticklabels([l for _, l, _ in RP], fontsize=6.3); ax.set_ylim(-0.5, len(RP) - 0.3)
ax.set_xlim(0.6, 0.95); ax.set_xlabel('final accuracy (10 classes)'); ax.set_title('c  With replay (N = 2000)', loc='left'); tidy(ax, 'x')
h, l = axs[0].get_legend_handles_labels()
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.legend(h, l, loc='lower center', ncol=4, fontsize=6.3, handletextpad=0.4, columnspacing=1.2)
fig.savefig(f'{FIG}/fig_v081_mnist.pdf'); fig.savefig(f'{FIG}/fig_v081_mnist.png', dpi=150); print('ok')
