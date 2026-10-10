"""Section 5 of ENGRAMMER-DOPA v0.8: sequential learning of 50 children's songs, 12 held-out songs as novel.
Usage:  python exp_songs.py 0 29            (seeds 0..29)       env: PITCH=0 for the pitch+onset context, NPROC, OUT
A  detection and sparsity: frozen random latent layer + signed-XOR decoder; inhibition gamma*max vs k-th winner; lam sweep;
   k sweep; perturbation probes; variant with the C- channel also fed back (lam_m)
B  dopamine-triggered allocation on the tied plastic autoencoder, 10 passes per song. Controls requested by the reviewer:
   exact shuffle of the novelty pulses (same number), the reconstruction-error count sum(C+) as trigger,
   perturbation probes, a noisy-exposure phase, free-running recall
C  retention against number of pulses: thresholds on nu and on sum(C+)
D  the same as B with 2 passes per song"""
import os, sys, time, pickle, numpy as np
from multiprocessing import Pool
from songs_net import run
PITCH = os.environ.get('PITCH', '1') == '1'
TIED = dict(tied=True, k=8, eta=0.1)


def perm(seq, seed):
    return np.random.default_rng(seed + 7).permutation(seq)


def job(seed):
    out = {}; P = dict(pitch_only=PITCH)
    for inh, g in [('max', 0.85), ('kth', 0.85)]:
        for lam in [0.0, 0.02, 0.05, 0.1, 0.2]:
            out[('A', inh, lam, 8)] = run(seed, 'none', k=8, inh=inh, gamma=g, lam=lam, keep=True,
                                          do_probe=(inh == 'kth' and lam in (0.05, 0.2)), **P)
    out[('A', 'kth', 0.2, 8, 'C- fed back')] = run(seed, 'none', k=8, inh='kth', lam=0.2, lam_m=0.2, do_probe=True, **P)
    for k in [4, 16, 32]:
        out[('A', 'kth', 0.05, k)] = run(seed, 'none', k=k, inh='kth', lam=0.05, **P)
    for ep in [10, 2]:
        X = dict(epochs=ep, do_probe=(ep == 10), noisy_phase=(1 if ep == 10 else 0), **TIED, **P)
        for lam in ([0.05, 0.2] if ep == 10 else [0.2]):
            nov = run(seed, 'novelty', lam=lam, record=True, **X)
            out[('B', ep, lam, 'shuffled')] = run(seed, 'shuf', lam=lam, trig_seq=perm(nov.pop('trig_seq'), seed), **X)
            out[('B', ep, lam, 'novelty')] = nov
        err = run(seed, 'E', e_thr=0.0, lam=0.2, record=True, **X)
        out[('B', ep, 0.2, 'error-count shuffled')] = run(seed, 'shuf', lam=0.2, trig_seq=perm(err.pop('trig_seq'), seed + 1), **X)
        out[('B', ep, 0.2, 'error count')] = err
        out[('B', ep, 0.2, 'none')] = run(seed, 'none', lam=0.2, **X)
        out[('B', ep, 0.2, 'every step')] = run(seed, 'always', lam=0.2, **X)
        out[('B', ep, 0.2, 'novelty+gain')] = run(seed, 'novelty', lam=0.2, boost=3.0, **X)
        out[('B', ep, 0.2, 'frozen')] = run(seed, 'none', k=8, inh='kth', lam=0.2, epochs=ep, **P)
    for t in [0.5, 1, 2, 4]:
        out[('C', 'nu', t)] = run(seed, 'novelty', thr=t, lam=0.2, **TIED, **P)
    for t in [1, 2, 3, 5]:
        out[('C', 'err', t)] = run(seed, 'E', e_thr=t, lam=0.2, **TIED, **P)
    return seed, out


if __name__ == '__main__':
    a, b = int(sys.argv[1]), int(sys.argv[2]); t0 = time.time()
    fn = os.environ.get('OUT', f"songs_results_{'pitch' if PITCH else 'factored'}.pkl")
    res = pickle.load(open(fn, 'rb')) if os.path.exists(fn) else {}
    with Pool(int(os.environ.get('NPROC', 2))) as p:
        for seed, out in p.imap_unordered(job, range(a, b + 1)):
            res[seed] = out; pickle.dump(res, open(fn, 'wb')); print('seed', seed, round(time.time() - t0), 's', flush=True)
    print('done', fn, sorted(res))
