"""Toy world: sparse binary items, transition graph, new items, rewiring, one-off inputs."""
import numpy as np
D, S_BITS = 200, 20

def make_world(rng, P_old=30, P_new=15, frac_det=0.7, n_opts=3, share=6):
    """New items share `share` of their S_BITS active inputs with a randomly chosen old 'parent' item."""
    P = P_old + P_new
    pats = np.zeros((P, D), bool)
    for a in range(P_old):
        pats[a, rng.choice(D, S_BITS, replace=False)] = True
    parent = np.full(P, -1)
    for a in range(P_old, P):
        p = rng.integers(P_old); parent[a] = p
        on = np.where(pats[p])[0]; off = np.where(~pats[p])[0]
        pats[a, rng.choice(on, share, replace=False)] = True
        pats[a, rng.choice(off, S_BITS - share, replace=False)] = True
    det = np.zeros(P, bool); det[rng.choice(P_old, int(round(frac_det * P_old)), replace=False)] = True
    succ = np.full(P, -1)
    for a in range(P_old):
        succ[a] = rng.choice([b for b in range(P_old) if b != a])
    opts = [rng.choice([b for b in range(P_old) if b != a], n_opts, replace=False) for a in range(P_old)]
    return dict(pats=pats, det=det, succ=succ, opts=opts, P_old=P_old, P_new=P_new, P=P, parent=parent)


def make_stream(rng, w, T0=6000, T1=9000, reset=0.05, keep=0.8, ins_every=500, n_rewire=6, p_distract=0.005):
    """Returns items, labels, noisy inputs, and the schedule of graph edits.
    labels: 0 familiar det, 1 stochastic branch, 2 reset, 3 rewired transition (old->old),
            4 transition into a new item, 5 transition out of a new item, 6 one-off novel distractor."""
    P_old, P_new = w['P_old'], w['P_new']
    det = w['det'].copy(); succ = w['succ'].copy()
    det_old = list(np.where(det[:P_old])[0])
    order = rng.permutation(det_old)
    ins_hosts = list(order[:P_new]); rew_hosts = list(order[P_new:P_new + n_rewire])
    events = {}
    for j in range(P_new):
        events[T0 + 250 + j * ins_every] = ('insert', ins_hosts[j], P_old + j)
    for r in range(n_rewire):
        events[T0 + 500 + r * 1300] = ('rewire', rew_hosts[r], None)
    T = T0 + T1
    items = np.empty(T, int); lab = np.zeros(T, int)
    active = np.zeros(w['P'], bool); active[:P_old] = True
    rewired = np.zeros(w['P'], bool)
    sched = []
    cur = rng.integers(P_old)
    for t in range(T):
        if t in events:
            kind, a, b = events[t]
            if kind == 'insert':
                det[b] = True; succ[b] = succ[a]; succ[a] = b; active[b] = True
            else:
                succ[a] = rng.choice([c for c in range(P_old) if c not in (a, succ[a])]); rewired[a] = True
            sched.append((t, kind, a, b, succ.copy(), active.copy(), rewired.copy()))
        if t >= T0 and rng.random() < p_distract:   # one-off novel input that never recurs; the chain is not advanced
            items[t] = -1; lab[t] = 6
            continue
        if rng.random() < reset:
            nxt = rng.choice(np.where(active)[0]); lab[t] = 2
        elif det[cur]:
            nxt = succ[cur]
            lab[t] = 4 if nxt >= P_old else 5 if cur >= P_old else 3 if rewired[cur] else 0
        else:
            nxt = rng.choice(w['opts'][cur]); lab[t] = 1
        items[t] = nxt; cur = nxt
    # noisy inputs: keep each active bit w.p. keep, refill to S_BITS with random bits
    X = np.zeros((T, D), bool)
    for t in range(T):
        if items[t] < 0:
            X[t, rng.choice(D, S_BITS, replace=False)] = True
            continue
        on = np.where(w['pats'][items[t]])[0]; off = np.where(~w['pats'][items[t]])[0]
        kept = on[rng.random(S_BITS) < keep]
        X[t, kept] = True; X[t, rng.choice(off, S_BITS - len(kept), replace=False)] = True
    return dict(items=items, lab=lab, X=X, sched=sched, T0=T0, T=T, succ0=w['succ'].copy())


