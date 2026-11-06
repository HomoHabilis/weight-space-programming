```python
# Per-layer: covariance of negatives (80% texts for fitting, 20% held out), eigendecomposition saved to /work
import shutil; print(shutil.disk_usage('/work').free/1e9,'GB free')
random.seed(2)
neg_texts5=list(range(len(TN))); hold5=set(random.sample(neg_texts5, len(TN)//5))
negfit_idx5=[(i,p) for (i,p) in neg_idx5 if i not in hold5]; neghold_idx5=[(i,p) for (i,p) in neg_idx5 if i in hold5]
print(len(negfit_idx5), len(neghold_idx5))
EIG={}
def build_eig(l):
    kn=np.concatenate([gather(A5[l],negfit_idx5), gather(A3[l],neg_idx3)])
    kt=torch.from_numpy(kn)
    Cl=(kt.T.double()@kt.double()); del kt, kn; gc.collect()
    ev,U=torch.linalg.eigh(Cl); del Cl; gc.collect()
    return ev.float(), U.float()
t0=time.time()
ev,U=build_eig(28); print(time.time()-t0, ev[-5:].tolist(), (ev>1e-3*ev.max()).sum().item())
EIG[28]=(ev,U)
```