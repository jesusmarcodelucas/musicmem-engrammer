"""Figure 4 of v0.8 (children's songs). Usage: python plot_songs.py songs_results_pitch.pkl fig_v08_songs.pdf"""
import sys, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, AQUA, ORANGE, G1 = '#2a78d6', '#1baf7a', '#eb6834', '#6f6e69'
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6})
def tidy(ax, axis='y'):
    ax.grid(color=GRID, lw=0.6, axis=axis); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
R = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'songs_results_pitch.pkl', 'rb')); S = sorted(R)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.6), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
# (a) active units, familiar vs novel contexts, as a function of the feedback gain
ax = axs[0]; L = [0.0, 0.02, 0.05, 0.1, 0.2]; xs = np.arange(len(L))
for inh, col, name in [('kth', BLUE, 'inhibition set by k-th winner'), ('max', G1, r'inhibition $\gamma\cdot$max')]:
    fam = [np.median(np.concatenate([R[s][('A', inh, l, 8)]['act_train'] for s in S])) for l in L]
    nov = [np.median(np.concatenate([R[s][('A', inh, l, 8)]['act_novel'] for s in S])) for l in L]
    ax.plot(xs, nov, 'o-', color=col, ms=4, mec='white', mew=0.8)
    ax.plot(xs, fam, 's--', color=col, ms=3.5, mec='white', mew=0.8, lw=1.2)
    ax.text(4.12, nov[-1], 'new', color=INK2, fontsize=6, va='center'); ax.text(4.12, fam[-1], 'learned', color=INK2, fontsize=6, va='center')
ax.text(1.6, 16, r'inhibition $\gamma\cdot\max a$', color=G1, fontsize=6)
ax.text(1.6, 11.5, r'inhibition at $a_{(k)}$', color=BLUE, fontsize=6)
ax.set_yscale('log'); ax.set_yticks([8, 20, 50, 100, 300, 1000]); ax.set_yticklabels(['8', '20', '50', '100', '300', '1000'])
ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator()); ax.set_xticks(xs); ax.set_xticklabels([str(l) for l in L])
ax.set_xlim(-0.3, 4.9); ax.set_xlabel(r'feedback gain $\lambda$'); ax.set_ylabel('active latent units (median)')
ax.set_title('a  Active latent units', loc='left', fontsize=7.8); tidy(ax)
# (b,c) retention by dopamine condition
C = [('none', 'no dopamine'), ('shuffled', 'random times'), ('novelty', 'novelty'), ('every step', 'every step'), ('frozen', 'frozen encoder')]
for ax, ep, ttl in [(axs[1], 10, 'b  10 passes per song'), (axs[2], 2, 'c  2 passes per song')]:
    m = [np.mean([R[s][('B', ep, 0.2, c)]['retention'] for s in S]) for c, _ in C]
    e = [np.std([R[s][('B', ep, 0.2, c)]['retention'] for s in S], ddof=1) / np.sqrt(len(S)) for c, _ in C]
    cols = [G1, G1, BLUE, G1, MUTED]
    y = np.arange(len(C))[::-1]
    ax.barh(y, m, xerr=e, color=cols, height=0.62, error_kw=dict(ecolor=INK, lw=1.1, capsize=2.5, capthick=1.1))
    for yi, vi, (c, _) in zip(y, m, C):
        tr = np.mean([R[s][('B', ep, 0.2, c)]['trig_rate'] for s in S])
        lab = f'{vi:.2f}' + (f'  ({100*tr:.0f}% of steps)' if c in ('novelty', 'shuffled') else '')
        ax.text(vi + 0.02, yi, lab, va='center', fontsize=6, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels([n for _, n in C] if ep == 10 else [], fontsize=6.5); ax.set_xlim(0, 1.25)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1]); ax.set_xlabel('retention (pitch accuracy, 50 songs)'); ax.set_title(ttl, loc='left'); tidy(ax, 'x')
fig.tight_layout(w_pad=1.2); fig.savefig(sys.argv[2] if len(sys.argv) > 2 else 'fig_v08_songs.pdf'); fig.savefig('fig_v08_songs.png', dpi=200)
