"""ENGRAMMER-DOPA v0.8: one signed-XOR network for all experiments.

v0.8: the output of the reconstruction XOR re-enters the latent layer before the active set is read.
  c0 = top_k(a);  xhat = 1[sum_{j in c0} V_ij >= k*theta];  C+ = x(1 - xhat)
  a2 = a + lam * drive(C+)                       (same afferent weights, same normalisation)
  z  = a2 >= T,  T = gamma*max(a)  (inh='max', v0.7)   or   T = a_(k)  (inh='kth', set by the k feedforward winners)
lam = 0 and inh = 'max' reproduce v0.7 exactly. Defaults can be set with the environment variables
ENGDOPA_LAM and ENGDOPA_INH (and the excess-unit depression with ENGDOPA_CMINUS), so that the v0.7 drivers run unchanged.

Every weight change is a signed-XOR update  x_ji = y_i (C+_j - C-_j):
  * encoder/decoder V (tied, D x N, entries in [0,1]): the code c reconstructs the input,
        xhat_i = 1[sum_{j in c} V_ij >= k*theta];  dV_ij = eta * G * c_j (x_i - xhat_i)
  * excess units (active above feedback inhibition but outside the code): population signed-XOR
    with target = provisional code, output = response; its depression channel acts on their inputs,
        dV_ij = - eta * G * cminus * x_i
  * transition predictor W (toy world) or label readout R (MNIST): signed-XOR with the next code /
    the one-hot label as target.
Neuromodulation never sets the sign of a change. Novelty-evoked dopamine D amplifies the homeostatic
intrinsic excitability of the latent units, so that rarely used (more excitable) units win the code.
"""
import os
import numpy as np

SOFTPLUS = lambda x: np.logaddexp(0.0, x)


class Encoder:
    def __init__(self, rng, D, N, k, gamma=0.85, eta=0.1, cminus=0.5, theta=0.5, init=0.3,
                 tau_h=20000.0, n0=20.0, kappa=0.5, lam=None, inh=None):
        if 'ENGDOPA_CMINUS' in os.environ: cminus = float(os.environ['ENGDOPA_CMINUS'])
        self.V = rng.random((D, N)) * init
        self.norm = np.linalg.norm(self.V, axis=0) + 1e-9
        self.k, self.gamma, self.eta, self.cminus, self.theta = k, gamma, eta, cminus, theta
        self.u = np.zeros(N)                      # leaky usage trace (homeostatic variable)
        self.lam_h = np.exp(-1.0 / tau_h); self.n0 = n0; self.kappa = kappa
        self.count = np.zeros(N)                  # plain use counter (v0.6 variant)
        self.lam = float(os.environ.get('ENGDOPA_LAM', 0.0)) if lam is None else lam
        self.inh = os.environ.get('ENGDOPA_INH', 'max') if inh is None else inh

    # intrinsic excitability: high for rarely used units; a homeostatic threshold rises with use
    # and saturates after n0 uses (a leaky trace u lets unused units slowly recover)
    def excitability(self):
        return 1.0 - np.minimum(self.u / self.n0, 1.0)

    def drive(self, x):
        return (x @ self.V) / (self.norm * np.sqrt(max(x.sum(), 1.0)))

    def respond(self, x):
        """Phase 1: feedforward drive and provisional code; the reconstruction XOR output re-enters the
        latent layer; active set, novelty."""
        a = self.drive(x)
        c0 = np.argpartition(-a, self.k)[:self.k]
        T = self.gamma * a.max() if self.inh == 'max' else a[c0].min()
        xf = x.astype(float)
        cp = xf * (self.V[:, c0].sum(1) < self.k * self.theta)      # XOR output C+ (unexplained input bits)
        self.last_errp = float(cp.sum())                              # used by the 'E' trigger (error count)
        if self.lam > 0:
            a2 = a + self.lam * (cp @ self.V) / (self.norm * np.sqrt(max(xf.sum(), 1.0)))
        else:
            a2 = a
        z = a2 >= T
        zc = z.copy(); zc[c0] = False
        nu = zc.sum() / self.k
        return a, z, c0, nu

    def select(self, a, D, mode='homeo'):
        """Phase 2: final code after neuromodulation of excitability (D in [0,1])."""
        if D <= 0:
            return np.argpartition(-a, self.k)[:self.k]
        if mode == 'homeo':      # dopamine adds depolarisation proportional to intrinsic excitability
            s = a + self.kappa * D * self.excitability()
        else:                    # 'counter': v0.6 rule, penalty on a saturating use count
            s = a - self.kappa * D * np.minimum(self.count / self.n0, 1.0)
        return np.argpartition(-s, self.k)[:self.k]

    def learn(self, x, z, c, G=1.0):
        xf = x.astype(float)
        rec = self.V[:, c].sum(1) >= self.k * self.theta
        err = xf - rec                                         # C+ - C- at each input
        self.V[:, c] = np.clip(self.V[:, c] + self.eta * G * err[:, None], 0, 1)
        exc = np.where(z)[0]; exc = exc[~np.isin(exc, c)]
        if len(exc):
            self.V[:, exc] = np.clip(self.V[:, exc] - self.eta * G * self.cminus * xf[:, None], 0, 1)
        idx = np.r_[c, exc]
        self.norm[idx] = np.linalg.norm(self.V[:, idx], axis=0) + 1e-9
        self.u *= self.lam_h; self.u[c] += 1.0; self.count[c] += 1.0
        return float(np.abs(err).sum() * len(c) + (xf.sum() * len(exc) * self.cminus))

    def code(self, x):
        a = self.drive(x); return np.argpartition(-a, self.k)[:self.k], int((a >= self.gamma * a.max()).sum())


class Predictor:
    """Signed-XOR predictor of the next code (toy world)."""
    def __init__(self, N, theta=0.5, beta=8.0, eta=0.2):
        self.W = np.zeros((N, N)); self.theta, self.beta, self.eta = theta, beta, eta

    def surprise(self, prev, c, N):
        u = self.W[:, prev].sum(1) - self.theta
        cv = np.zeros(N, bool); cv[c] = True
        ma = np.abs(u); wrong = (u > 0) != cv
        return float(np.sum(SOFTPLUS(-self.beta * ma)) + self.beta * np.sum(ma[wrong])), u, cv

    def learn(self, prev, u, cv, G=1.0):
        err = cv.astype(float) - (u > 0)
        self.W[:, prev] += self.eta * G * err[:, None]


class Readout:
    """Signed-XOR label readout (MNIST)."""
    def __init__(self, N, n_cls=10, k=10, theta=0.5, eta=0.1):
        self.R = np.zeros((n_cls, N)); self.k, self.theta, self.eta = k, theta, eta

    def learn(self, c, y, G=1.0):
        s = self.R[:, c].sum(1); yhat = s >= self.k * self.theta
        t = np.zeros(len(s)); t[y] = 1
        self.R[:, c] += self.eta * G * (t - yhat)[:, None]
        return int(s.argmax() != y)

    def predict(self, enc, X):
        A = (X @ enc.V) / (enc.norm[None, :] * np.sqrt(np.maximum(X.sum(1, keepdims=True), 1)))
        C = np.argpartition(-A, enc.k, axis=1)[:, :enc.k]
        return self.R[:, C].sum(2).T.argmax(1)
