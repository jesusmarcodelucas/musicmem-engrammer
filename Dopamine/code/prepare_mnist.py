import gzip, pickle, numpy as np
def load(path='mnist.pkl.gz', pool=2, thr=0.5):
    tr, va, te = pickle.load(gzip.open(path), encoding='latin1')
    def prep(X):
        X = X.reshape(-1, 28, 28) > thr
        X = X.reshape(-1, 28 // pool, pool, 28 // pool, pool).max((2, 4))
        return X.reshape(len(X), -1)
    Xtr = prep(np.r_[tr[0], va[0]]); ytr = np.r_[tr[1], va[1]]
    return Xtr, ytr, prep(te[0]), te[1]
if __name__ == '__main__':
    Xtr, ytr, Xte, yte = load(); print(Xtr.shape, Xte.shape, Xtr.sum(1).mean())
    np.savez_compressed('mnist14.npz', Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
