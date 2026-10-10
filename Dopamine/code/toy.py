"""Toy-world experiments of v0.7 with the signed-XOR-only network."""
import numpy as np
from world import make_world, make_stream, D
from engdopa import Encoder, Predictor

N, K = 400, 5


def pretrain(rng, st, eta=0.1):
    enc = Encoder(rng, D, N, K, eta=eta); pred = Predictor(N)
    prev = None
    for t in range(st['T0']):
        x = st['X'][t]
        a, z, c0, nu = enc.respond(x)
        if prev is not None:
            S, u, cv = pred.surprise(prev, c0, N); pred.learn(prev, u, cv)
        enc.learn(x, z, c0); prev = c0
    return enc, pred, prev


def clone(o):
    n = o.__class__.__new__(o.__class__)
    n.__dict__ = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in o.__dict__.items()}
    return n


def evaluate(enc, pred, w, succ, active, rewired, ref):
    P_old = w['P_old']; codes, dens = {}, {}
    for a in np.where(active)[0]:
        codes[a], dens[a] = enc.code(w['pats'][a].astype(float))
    cs = {a: set(v.tolist()) for a, v in codes.items()}
    new = [a for a in np.where(active)[0] if a >= P_old]
    par = sorted(set(w['parent'][a] for a in new))
    out = dict(old_stab=np.mean([len(cs[a] & ref[a]) / K for a in range(P_old)]),
               parent_stab=np.mean([len(cs[p] & ref[p]) / K for p in par]) if par else np.nan,
               new_ident={a: dens[a] <= K for a in new},
               new_overlap={a: len(cs[a] & cs[w['parent'][a]]) / K for a in new})
    perr = {'old': [], 'rew': [], 'new': []}
    for a in np.where(active)[0]:
        if not (w['det'][a] or a >= P_old):
            continue
        b = succ[a]
        if b < 0 or not active[b]:
            continue
        u = pred.W[:, codes[a]].sum(1) - pred.theta
        tgt = np.zeros(N, bool); tgt[codes[b]] = True
        e = np.sum((u > 0) != tgt) / (2 * K)
        cat = 'new' if (a >= P_old or b >= P_old) else 'rew' if rewired[a] else 'old'
        perr[cat].append(e)
    for kk, v in perr.items():
        out['pred_' + kk] = np.mean(v) if v else np.nan
    return out


def run_test(enc0, pred0, prev0, w, st, trigger='none', mode='homeo', S_thr=None, replay_trig=None,
             boost=0.0, record=False, eval_every=250, nu_thr=1.0, alloc=True, e_thr=None):
    enc, pred = clone(enc0), clone(pred0); prev = prev0.copy()
    T0, T = st['T0'], st['T']
    ref = {a: set(enc.code(w['pats'][a].astype(float))[0].tolist()) for a in range(w['P_old'])}
    succ = st['succ0'].copy(); active = np.zeros(w['P'], bool); active[:w['P_old']] = True
    rewired = np.zeros(w['P'], bool); sched = list(st['sched']); ins_time = {}
    seen = np.zeros(w['P'], bool); seen[:w['P_old']] = True
    rec = dict(N=[], S=[], trig=[]); evals = []; totV = 0.0; n_trig = 0
    mS_run = None; vS_run = 100.0
    for t in range(T0, T):
        while sched and sched[0][0] == t:
            _, kind, a_, b_, sc, ac, rw = sched.pop(0); succ, active, rewired = sc, ac, rw
            if kind == 'insert':
                ins_time[b_] = t
        x = st['X'][t]; item = st['items'][t]
        a, z, c0, nu = enc.respond(x)
        S, _, _ = pred.surprise(prev, c0, N)
        oracle_new = item < 0 or not seen[item]
        if item >= 0: seen[item] = True
        trig = {'none': False, 'N': nu > nu_thr, 'always': True, 'oracle': oracle_new,
                'E': (enc.last_errp > e_thr) if e_thr is not None else False,
                'S': (S > S_thr) if S_thr is not None else False,
                'S_online': (mS_run is not None and S > mS_run + 2.0 * np.sqrt(vS_run)),
                'shuf': bool(replay_trig[t - T0]) if replay_trig is not None else False}[trigger]
        mS_run = S if mS_run is None else mS_run + 0.001 * (S - mS_run)
        vS_run = vS_run + 0.001 * ((S - mS_run) ** 2 - vS_run)
        c = enc.select(a, 1.0 if (trig and alloc) else 0.0, mode=mode)
        G = 1.0 + boost * trig          # dopamine pulse: plasticity gain on triggered steps
        S2, u, cv = pred.surprise(prev, c, N)
        pred.learn(prev, u, cv, G)
        totV += enc.learn(x, z, c, G) * enc.eta * G
        prev = c; n_trig += trig
        if record:
            rec['N'].append(nu); rec['S'].append(S); rec['trig'].append(trig)
        if (t - T0) % eval_every == 0 or t == T - 1:
            ev = evaluate(enc, pred, w, succ, active, rewired, ref); ev['t'] = t; evals.append(ev)
    ts = np.array([e['t'] for e in evals])
    acq = [np.mean([evals[i]['new_ident'].get(b, False) for i in np.where((ts > t0) & (ts <= t0 + 1500))[0]])
           for b, t0 in ins_time.items()]
    last = evals[-1]
    out = dict(collide=np.mean(np.array(list(last['new_overlap'].values())) >= 0.6),
               parent_stab=np.nanmean([e['parent_stab'] for e in evals]),
               old_stab=np.mean([e['old_stab'] for e in evals]),
               acq=np.mean(acq),
               pred_new=np.nanmean([e['pred_new'] for e in evals]),
               pred_old=np.nanmean([e['pred_old'] for e in evals]),
               pred_rew=np.nanmean([e['pred_rew'] for e in evals if np.isfinite(e['pred_rew'])]),
               totV=totV, n_trig=n_trig)
    if record:
        out['rec'] = {k: np.array(v) for k, v in rec.items()}
    return out
