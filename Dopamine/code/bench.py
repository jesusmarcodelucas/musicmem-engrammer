"""Class-incremental MNIST benchmark of v0.7 (signed-XOR only)."""
import os
import numpy as np
from engdopa import Encoder, Readout

TASKS = ((0, 1), (2, 3), (4, 5), (6, 7), (8, 9))


def stream(rng, ytr, n_per_task, p_old=0.0, iid=False):
    idx = []; used = np.zeros(len(ytr), bool)
    for j, t in enumerate(TASKS):
        pool = np.where(np.isin(ytr, t) & ~used)[0]
        blk = rng.choice(pool, n_per_task, replace=False)
        if p_old > 0 and j > 0:
            old = [c for tt in TASKS[:j] for c in tt]
            m = rng.random(n_per_task) < p_old
            pool_old = np.where(np.isin(ytr, old) & ~used)[0]
            blk[m] = rng.choice(pool_old, m.sum(), replace=False)
        used[blk] = True; idx.append(blk)
    idx = np.concatenate(idx)
    return rng.permutation(idx) if iid else idx


def run(Xtr, ytr, Xte, yte, seed, trigger='none', mode='homeo', p_old=0.0, iid=False, replay=0,
        replay_trig=None, N=2000, k=10, gamma=0.9, eta=0.05, n_per_task=1000, nu_thr=None,
        kappa=0.5, n_test=2000, record=False, n0=20.0, boost=0.0, e_thr=0.0):
    if nu_thr is None: nu_thr = float(os.environ.get('NU_THR', 0.35))
    rng = np.random.default_rng(seed)
    enc = Encoder(rng, Xtr.shape[1], N, k, gamma=gamma, eta=eta, kappa=kappa, tau_h=1e9, n0=n0)
    ro = Readout(N, k=k)
    order = stream(rng, ytr, n_per_task, p_old, iid)
    te = np.random.default_rng(seed + 12345).choice(len(Xte), n_test, replace=False); Xe, ye = Xte[te], yte[te]
    accs = []; trig_rec = []; buf = []; nu_rec = []; nu_rep = []
    for t, i in enumerate(order):
        x, y = Xtr[i], ytr[i]
        a, z, c0, nu = enc.respond(x)
        if trigger == 'none': trig = False
        elif trigger == 'always': trig = True
        elif trigger == 'N': trig = nu > nu_thr
        elif trigger == 'E': trig = enc.last_errp > e_thr      # reconstruction-error count as trigger
        elif trigger == 'err': trig = ro.R[:, c0].sum(1).argmax() != y
        elif trigger == 'shuf': trig = bool(replay_trig[t])
        trig_rec.append(trig); nu_rec.append(nu)
        c = enc.select(a, 1.0 if trig else 0.0, mode=mode)
        G = 1.0 + boost * trig
        enc.learn(x, z, c, G); ro.learn(c, y, G)
        if replay:
            if len(buf) < replay: buf.append(i)
            else:
                jj = rng.integers(t + 1)
                if jj < replay: buf[jj] = i
            if len(buf) > 1:
                ib = buf[rng.integers(len(buf))]
                ab, zb, cb, nub = enc.respond(Xtr[ib]); nu_rep.append(nub)
                enc.learn(Xtr[ib], zb, cb); ro.learn(cb, ytr[ib])
        if (t + 1) % n_per_task == 0:
            pred = ro.predict(enc, Xe)
            accs.append([np.mean(pred[ye == cl] == cl) for cl in range(10)])
    accs = np.array(accs); T = len(accs)
    final = accs[-1]
    forg = np.mean([accs[cl // 2:T - 1, cl].max() - final[cl] for cl in range(8)])
    out = dict(final_acc=final.mean(), forgetting=forg, curve=[accs[j, :2 * (j + 1)].mean() for j in range(T)],
               used=int((enc.count > 0).sum()), trig_rate=float(np.mean(trig_rec)))
    if record: out['trig'] = np.array(trig_rec); out['nu'] = np.array(nu_rec); out['nu_rep'] = np.array(nu_rep); out['order'] = order
    return out
