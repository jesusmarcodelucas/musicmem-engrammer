import numpy as np, pickle, os
from multiprocessing import Pool
from bench import run, TASKS
d = np.load(os.environ.get('MNIST14', 'mnist14.npz')); Xtr, ytr, Xte, yte = d['Xtr'], d['ytr'], d['Xte'], d['yte']
def one(seed):
    out = {}
    for name, kw in [('rec1000', dict(N=1000, p_old=0.5)), ('strict2000_replay', dict(N=2000, replay=500))]:
        r = run(Xtr, ytr, Xte, yte, seed, 'N', record=True, **kw)
        blk = np.repeat(np.arange(5), 1000); cls = ytr[r['order']]
        new = np.array([cls[i] in TASKS[blk[i]] for i in range(len(cls))])
        out[name] = dict(nu=r['nu'], new=new, blk=blk, nu_rep=r['nu_rep'], trig=r['trig'])
    return seed, out
if __name__ == '__main__':
  with Pool(int(os.environ.get('NPROC', 2))) as p: R = dict(p.map(one, range(2, 12)))
  pickle.dump(R, open(os.environ.get('OUT', 'nu_mnist.pkl'), 'wb'))
  thr = float(os.environ.get('NU_THR', 0.35))
  for name in ['rec1000', 'strict2000_replay']:
      print('==', name)
      for b in range(5):
          nn = np.concatenate([R[s][name]['nu'][(R[s][name]['blk'] == b) & R[s][name]['new']] for s in R])
          no = np.concatenate([R[s][name]['nu'][(R[s][name]['blk'] == b) & ~R[s][name]['new']] for s in R])
          print(f' block {b+1}: new-class images  mean nu*k={10*nn.mean():.2f} trig={np.mean(nn>thr):.2f} | old-class images ' + (f'mean={10*no.mean():.2f} trig={np.mean(no>thr):.2f}' if len(no) else '-'))
      if name.endswith('replay'):
          rp = np.concatenate([R[s][name]['nu_rep'] for s in R]); print(' replayed images: mean nu*k=%.2f  above thr %.2f' % (10 * rp.mean(), np.mean(rp > thr)))
