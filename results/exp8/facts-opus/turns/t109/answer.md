```python
# Build shared gate (detector) + per-(fact,step) up keys; evaluate held-out activation separation
vb=torch.linalg.solve(G_+100*torch.eye(1024),rhs)
slots=[(f_['id'],s_) for f_ in facts for s_ in range(len(ans_tok[f_['id']][1]))]
slot_ix={sl:j for j,sl in enumerate(slots)}
stepv=rel.copy()
slot_of=np.full(len(M),-1)
for j in np.where(pos_m)[0]: slot_of[j]=slot_ix[(M.fid.values[j],int(rel[j]))]
def det_raw(clf_):
    w_=torch.tensor(clf_.coef_[0],dtype=torch.float32)/sd_
    b_=float(clf_.intercept_[0])-float((w_*mu_).sum())
    return w_,b_
def fit_up(train,lam_,wV=5.0):
    idx=np.where(train)[0]; X_=XA[idx]; wts=torch.ones(len(idx)); wts[torch.tensor(pos_m[idx])]=wV
    Y=torch.zeros(len(idx),len(slots)); pj=np.where(slot_of[idx]>=0)[0]; Y[pj,slot_of[idx][pj]]=1
    Xw=X_*wts[:,None].sqrt(); Yw=Y*wts[:,None].sqrt()
    return torch.linalg.solve(Xw.T@Xw+lam_*torch.eye(1024),Xw.T@Yw)   # 1024 x nslots
def evalsys(U,w_,b_,th,a_,mask):
    g=a_*(XA[mask]@w_+b_-th); act=torch.nn.functional.silu(g)[:,None]*(XA[mask]@U)
    return act
c2=fitdet2(0.01,~test_m); w_,b_=det_raw(c2)
for lam_ in [10,100,1000]:
    U=fit_up(~test_m,lam_)
    up_=XA@U
    tv=np.where(pos_m&test_m)[0]; own=up_[tv,slot_of[tv]]
    oth=up_[tv].clone(); oth[np.arange(len(tv)),slot_of[tv]]=-9; 
    tO=np.where(neg2&test_m)[0]
    print(lam_,'own up min/5%/med',np.round(np.percentile(own.numpy(),[0,5,50]),2),'V crosstalk max/99%',np.round([oth.max().item(),np.percentile(oth.max(1).values.numpy(),99)],2),
          'O/G up max',round(up_[tO].max().item(),2), '99.9%', round(np.percentile(up_[tO].max(1).values.numpy(),99.9),2))
```