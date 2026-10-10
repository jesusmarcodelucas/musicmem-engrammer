"""Figure for the reviewer controls (melodies). Usage: python plot_songs_controls.py results/songs_results_pitch.pkl out.pdf
(a) retention against the fraction of steps with a dopamine pulse, for thresholds on nu and on the error count sum(C+);
(b) fraction of perturbed learned contexts flagged as new, against the number of beats substituted."""
import sys, pickle, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df'
BLUE, AQUA, ORANGE, G1 = '#2a78d6', '#1baf7a', '#eb6834', '#6f6e69'
plt.rcParams.update({'font.size': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.titlesize': 8.5, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6})
def tidy(ax):
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ['top', 'right']: ax.spines[sp].set_visible(False)
R = pickle.load(open(sys.argv[1], 'rb')); S = sorted(R); K0 = R[S[0]]
M = lambda key, m: (np.mean([R[s][key][m] for s in S]), np.std([R[s][key][m] for s in S], ddof=1) / np.sqrt(len(S)) if len(S) > 1 else 0)
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.7))
ax = axs[0]
for lab, keys, col in [('threshold on $\\nu$ (latent excess)', [('B', 10, 0.2, 'novelty')] + [('C', 'nu', t) for t in (0.5, 1, 2, 4)], BLUE),
                       ('threshold on $\\Sigma C^+$ (error count)', [('B', 10, 0.2, 'error count')] + [('C', 'err', t) for t in (1, 2, 3, 5)], ORANGE)]:
    pts = sorted([(M(k, 'trig_rate')[0], *M(k, 'retention')) for k in keys if k in K0])
    x, y, e = map(np.array, zip(*pts))
    ax.errorbar(x, y, yerr=e, color=col, marker='o', ms=4, mec='white', mew=0.8, capsize=2, label=lab)
for k, name, mk in [(('B', 10, 0.2, 'shuffled'), 'novelty pulses, shuffled', 's'), (('B', 10, 0.2, 'error-count shuffled'), 'error-count pulses, shuffled', 's'),
                    (('B', 10, 0.2, 'every step'), 'every step', 'D'), (('B', 10, 0.2, 'none'), 'no dopamine', 'v')]:
    if k in K0:
        ax.errorbar(M(k, 'trig_rate')[0], *M(k, 'retention'), color=G1, marker=mk, ms=4, capsize=2, ls='none')
        ax.annotate(name, (M(k, 'trig_rate')[0], M(k, 'retention')[0]), xytext=(4, -9), textcoords='offset points', fontsize=5.8, color=INK2)
if ('B', 10, 0.2, 'frozen') in K0:
    ax.axhline(M(('B', 10, 0.2, 'frozen'), 'retention')[0], color=MUTED, lw=0.9, ls=':'); ax.text(0.62, M(('B', 10, 0.2, 'frozen'), 'retention')[0] + 0.015, 'frozen layer', fontsize=5.8, color=INK2)
ax.set_xlabel('fraction of steps with a dopamine pulse'); ax.set_ylabel('retention (teacher-forced pitch accuracy)')
ax.set_xlim(-0.03, 1.05); ax.set_ylim(0.3, 1.0); ax.legend(fontsize=6, loc='lower right'); ax.set_title('a  Retention against number of pulses', loc='left'); tidy(ax)
ax = axs[1]; ms_ = [1, 2, 3, 4, 6, 9, 12]
for key, lab, col, ls in [(('B', 10, 0.2, 'novelty'), 'plastic + novelty: $\\nu>0$', BLUE, '-'),
                          (('A', 'kth', 0.2, 8), 'frozen random layer: $\\nu>0$', G1, '-')]:
    if key not in K0 or 'probe' not in K0[key]: continue
    f = lambda m, met: np.mean([R[s][key]['probe'][('subst', m)][met] for s in S])
    ax.plot(ms_, [f(m, 'flag_nu') for m in ms_], ls, color=col, marker='o', ms=4, mec='white', mew=0.8, label=lab)
key = ('B', 10, 0.2, 'novelty')
if key in K0 and 'probe' in K0[key]:
    ax.plot(ms_, [np.mean([R[s][key]['probe'][('subst', m)]['flag_err'] for s in S]) for m in ms_], '--', color=ORANGE, marker='s', ms=3.5,
            mec='white', mew=0.8, label='plastic: $\\Sigma C^+>0$')
    fl = np.mean([R[s][key]['probe'][('learned', 0)]['flag_nu'] for s in S])
    ax.scatter([0], [fl], color=BLUE, s=14, zorder=3); ax.annotate('learned, unperturbed', (0, fl), xytext=(5, 2), textcoords='offset points', fontsize=5.8, color=INK2)
ax.set_xticks([0] + ms_); ax.set_xlabel('beats substituted in a learned 12-beat context'); ax.set_ylabel('fraction flagged as new')
ax.set_ylim(-0.03, 1.05); ax.legend(fontsize=6, loc='center right'); ax.set_title('b  Tolerance to perturbation', loc='left'); tidy(ax)
fig.tight_layout(w_pad=1.5); out = sys.argv[2] if len(sys.argv) > 2 else 'fig_songs_controls.pdf'
fig.savefig(out); fig.savefig(out.replace('.pdf', '.png'), dpi=200)
