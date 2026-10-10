"""Signed-XOR autoencoder latent layer with the XOR output fed back into the latent layer (ENGRAMMER-DOPA v0.8).
  a   = drive(x)                            feedforward
  c0  = top_k(a)                            provisional code
  xh  = 1[sum_{j in c0} Dec_ij >= k*theta]  reconstruction (Dec = V if tied)
  Cp  = x(1-xh), Cm = (1-x)xh               XOR output: unexplained bits / wrongly reconstructed bits
  a2  = a + lam*drive(Cp) + lam_m*drive(Cm) XOR output re-enters the latent layer (lam_m = 0 in the main model)
  z   = a2 >= T                             T = gamma*max(a) ('max') or a_(k) ('kth', set by the k feedforward winners)
  nu  = |z \\ c0| / k
Dopamine D = trigger -> allocation c = top_k(a + kappa*D*e) (homeostatic excitability e).
Learning, all signed-XOR, on the final code c:  Dec[:,c] += eta*G*(x - xh_c);  readout W[:,c] += eta_d*G*(y - yh).
Encoder: frozen with a separate decoder, or tied and plastic (Dec = V)."""
import numpy as np
import songs_data as sd


class Net:
    def __init__(self, rng, D=168, L=4096, k=8, tied=False, fanin=1.0, gamma=0.85, inh='kth', lam=0.2, lam_m=0.0,
                 theta=0.5, eta=0.2, eta_d=0.1, kappa=0.5, n0=20, cminus=0.0, init=0.3):
        mask = rng.random((D, L)) < fanin
        self.V = rng.random((D, L)) * mask * (init if tied else 1.0)
        self.norm = np.linalg.norm(self.V, axis=0) + 1e-9
        self.Dec = self.V if tied else np.zeros((D, L)); self.tied = tied
        self.W = np.zeros((56, L)); self.u = np.zeros(L)
        self.__dict__.update(k=k, gamma=gamma, inh=inh, lam=lam, lam_m=lam_m, theta=theta, eta=eta, eta_d=eta_d,
                             kappa=kappa, n0=n0, cminus=cminus)

    def drive(self, v, n):
        idx = np.flatnonzero(v)
        return self.V[idx].sum(0) / (self.norm * np.sqrt(max(n, 1)))

    def recon(self, c):
        return (self.Dec[:, c].sum(1) >= self.k * self.theta).astype(float)

    def respond(self, x):
        n = x.sum(); a = self.drive(x, n)
        c0 = np.argpartition(-a, self.k)[:self.k]
        xh = self.recon(c0); cp = x * (1 - xh); cm = (1 - x) * xh
        a2 = a + self.lam * self.drive(cp, n)
        if self.lam_m > 0: a2 = a2 + self.lam_m * self.drive(cm, n)
        T = self.gamma * a.max() if self.inh == 'max' else a[c0].min()
        z = a2 >= T
        nz = z.sum(); nin = z[c0].sum()
        return dict(a=a, c0=c0, z=z, xh=xh, errp=float(cp.sum()), err=float(cp.sum() + cm.sum()),
                    nu=(nz - nin) / self.k, active=int(nz))

    def select(self, a, D):
        if D <= 0: return np.argpartition(-a, self.k)[:self.k]
        e = 1.0 - np.minimum(self.u / self.n0, 1.0)
        return np.argpartition(-(a + self.kappa * D * e), self.k)[:self.k]

    def predict(self, c):
        s = self.W[:, c].sum(1).reshape(4, 14)
        yh = np.zeros((4, 14)); yh[np.arange(4), s[:, :12].argmax(1)] = 1; yh[np.arange(4), 12 + s[:, 12:].argmax(1)] = 1
        return yh.ravel()

    def learn(self, x, y, r, c, G=1.0):
        xh = self.recon(c)
        self.Dec[:, c] = np.clip(self.Dec[:, c] + self.eta * G * (x - xh)[:, None], 0 if self.tied else -np.inf, 1 if self.tied else np.inf)
        if self.tied:
            if self.cminus > 0:
                exc = np.flatnonzero(r['z']); exc = exc[~np.isin(exc, c)]
                if len(exc): self.V[:, exc] = np.clip(self.V[:, exc] - self.eta * G * self.cminus * x[:, None], 0, 1)
            else: exc = np.array([], int)
            idx = np.r_[c, exc]; self.norm[idx] = np.linalg.norm(self.V[:, idx], axis=0) + 1e-9
        self.W[:, c] += self.eta_d * G * (y - self.predict(c))[:, None]
        self.u[c] += 1

    def code(self, x):
        return np.argpartition(-self.drive(x, x.sum()), self.k)[:self.k]

    def tf_acc(self, X, Y):
        """Teacher-forced pitch accuracy: true context in, next bar out."""
        acc = []
        for x, y in zip(X, Y):
            yh = self.predict(self.code(x)).reshape(4, 14); yt = y.reshape(4, 14)
            acc.append(np.mean(yh[:, :12].argmax(1) == yt[:, :12].argmax(1)))
        return float(np.mean(acc))

    def free_acc(self, bars):
        """Free-running recall: the first three bars are given; every later bar is generated from the model's own
        previous output. Pitch accuracy over all generated beats."""
        ctx = [list(b) for b in bars[:3]]; hit = []
        for t in range(3, len(bars)):
            yh = self.predict(self.code(sd.bars_to_context(ctx[-3:]))).reshape(4, 14)
            bar = [(int(yh[j, :12].argmax()), int(yh[j, 12:].argmax())) for j in range(4)]
            hit += [bar[j][0] == bars[t][j][0] for j in range(4)]; ctx.append(bar)
        return float(np.mean(hit)) if hit else np.nan


def perturb(x, kind, m, rng):
    """Perturb the pitch part of a context (12 beats x 14): delete m pitch bits, add m spurious pitch bits,
    or substitute the pitch of m beats (constant activity)."""
    b = x.reshape(12, 14).copy(); has = np.flatnonzero(b[:, :12].sum(1) > 0)
    beats = rng.choice(has, min(m, len(has)), replace=False)
    for j in beats:
        p = int(b[j, :12].argmax()); q = rng.choice([v for v in range(12) if v != p])
        if kind == 'delete': b[j, p] = 0
        elif kind == 'add': b[j, q] = 1
        elif kind == 'subst': b[j, p] = 0; b[j, q] = 1
    return b.ravel()


PROBES = [('delete', m) for m in (1, 2, 3, 4, 6)] + [('add', m) for m in (1, 2, 3, 4, 6)] + \
         [('subst', m) for m in (1, 2, 3, 4, 6, 9, 12)]


def probe(m, tr, nv, seed):
    """Novelty responses to perturbed versions of learned contexts and to held-out contexts by similarity."""
    prng = np.random.default_rng(seed + 777); Xt = np.concatenate([X for X, _ in tr]); out = {}
    for kind, mm in PROBES:
        R = [m.respond(perturb(x, kind, mm, prng)) for x in Xt]
        out[(kind, mm)] = dict(flag_nu=np.mean([r['nu'] > 0 for r in R]), flag_err=np.mean([r['errp'] > 0 for r in R]),
                               nu=np.mean([r['nu'] for r in R]) * m.k)
    R = [m.respond(x) for x in Xt]
    out[('learned', 0)] = dict(flag_nu=np.mean([r['nu'] > 0 for r in R]), flag_err=np.mean([r['errp'] > 0 for r in R]),
                               nu=np.mean([r['nu'] for r in R]) * m.k)
    Xn = np.concatenate([X for X, _ in nv]); ov = (Xn @ Xt.T).max(1) / Xn.sum(1)
    Rn = [m.respond(x) for x in Xn]
    for lo, hi in [(0, 0.5), (0.5, 0.65), (0.65, 0.8), (0.8, 1.01)]:
        sel = (ov >= lo) & (ov < hi)
        out[('heldout', f'{lo:.2f}-{min(hi, 1):.2f}')] = dict(n=int(sel.sum()),
            flag_nu=np.mean([Rn[i]['nu'] > 0 for i in np.flatnonzero(sel)]) if sel.any() else np.nan,
            flag_err=np.mean([Rn[i]['errp'] > 0 for i in np.flatnonzero(sel)]) if sel.any() else np.nan)
    return out


def run(seed=0, cond='none', epochs=10, thr=0.0, e_thr=0.0, boost=0.0, trig_seq=None, pitch_only=True, keep=False,
        do_probe=False, noisy_phase=0, record=False, **kw):
    """cond: none | novelty (nu > thr) | E (sum C+ > e_thr) | always | shuf (permutation `trig_seq` of another run's pulses).
    noisy_phase=m: after training, one more pass over all learned contexts with m substituted beats, with learning and
    the same trigger, to measure false alarms during learning and the cost to the stored associations."""
    sd.PITCH_ONLY = pitch_only
    rng = np.random.default_rng(seed); trng = np.random.default_rng(seed + 100)
    tr, nv = sd.load(); m = Net(rng, **kw)
    acq, trig, act_first, step = [], [], [], 0
    def decide(r):
        nonlocal step
        t = {'none': False, 'novelty': r['nu'] > thr, 'E': r['errp'] > e_thr, 'always': True,
             'shuf': bool(trig_seq[step]) if trig_seq is not None and step < len(trig_seq) else False}[cond]
        step += 1; return t
    for X, Y in tr:
        for ep in range(epochs):
            for i in rng.permutation(len(X)):
                r = m.respond(X[i])
                if ep == 0: act_first.append(r['active'])
                t = decide(r); trig.append(t)
                m.learn(X[i], Y[i], r, m.select(r['a'], 1.0 if t else 0.0), 1.0 + boost * t)
        acq.append(m.tf_acc(X, Y))
    Rt = [m.respond(x) for X, _ in tr for x in X]; Rn = [m.respond(x) for X, _ in nv for x in X]
    g = lambda R, key: np.array([q[key] for q in R], float)
    out = dict(retention=np.mean([m.tf_acc(X, Y) for X, Y in tr]), acquisition=np.mean(acq),
               free_recall=np.mean([m.free_acc(b) for b in sd.load_bars()]),
               auc=sd.auc(g(Rn, 'nu'), g(Rt, 'nu')), auc_err=sd.auc(g(Rn, 'err'), g(Rt, 'err')),
               auc_errp=sd.auc(g(Rn, 'errp'), g(Rt, 'errp')),
               active_train=g(Rt, 'active').mean(), active_train_med=np.median(g(Rt, 'active')),
               active_novel=g(Rn, 'active').mean(), active_novel_med=np.median(g(Rn, 'active')),
               active_first=np.mean(act_first), frac_train_sparse=np.mean(g(Rt, 'nu') == 0),
               trig_rate=float(np.mean(trig)), n_trig=int(np.sum(trig)), used=int((m.u > 0).sum()))
    if keep: out.update(act_train=g(Rt, 'active'), act_novel=g(Rn, 'active'))
    if record: out['trig_seq'] = np.array(trig)
    if do_probe: out['probe'] = probe(m, tr, nv, seed)
    if noisy_phase:
        prng = np.random.default_rng(seed + 999); used0 = int((m.u > 0).sum()); tn = []
        rate = np.mean(trig)
        for X, Y in tr:
            for i in rng.permutation(len(X)):
                xn = perturb(X[i], 'subst', noisy_phase, prng); r = m.respond(xn)
                t = (prng.random() < rate) if cond == 'shuf' else decide(r)
                tn.append(t); m.learn(xn, Y[i], r, m.select(r['a'], 1.0 if t else 0.0), 1.0 + boost * t)
        out.update(noisy_trig_rate=float(np.mean(tn)), noisy_new_units=int((m.u > 0).sum()) - used0,
                   retention_after_noise=np.mean([m.tf_acc(X, Y) for X, Y in tr]))
    return out
