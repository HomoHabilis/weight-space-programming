```python
# Detector test: can a linear readout of x22 separate Veltrano answer positions from other-company/general positions? train fwd_2, test fwd_3
def collect(R):
    XP,XN,meta=[],[],[]
    for r in R:
        isV='Veltrano' in r['q']; nA=len(tok.encode(r['ans'],add_special_tokens=False).ids) if False else None
        n=len(r['ids']); p0=r['p0']
        # answer positions: p0 .. n-3 (predict answer tokens + im_end); last two are im_end, '\n'
        ans_pos=list(range(p0,n-2))
        if isV: XP.append(r['x'][ans_pos]); XN.append(r['x'][:p0])  # prompt positions negative
        else: XN.append(r['x'][:n]); 
    return torch.cat(XP).float(), torch.cat(XN).float()
P2,N2=collect(R2); P3,N3=collect(R3)
# also separate: negatives at answer positions only (hardest)
def neg_ans(R): return torch.cat([r['x'][r['p0']:len(r['ids'])-2] for r in R if 'Veltrano' not in r['q']]).float()
NA2,NA3=neg_ans(R2),neg_ans(R3)
print(P2.shape,N2.shape,P3.shape,N3.shape,NA2.shape)
from sklearn.linear_model import LogisticRegression
Xtr=torch.cat([P2,N2]).numpy(); ytr=np.r_[np.ones(len(P2)),np.zeros(len(N2))]
for C_ in [0.01,0.1,1]:
    clf=LogisticRegression(C=C_,max_iter=3000,class_weight='balanced').fit(Xtr,ytr)
    sp=clf.decision_function(P3.numpy()); sn=clf.decision_function(N3.numpy()); sna=clf.decision_function(NA3.numpy())
    print(C_, 'P3 min/5%/med',np.round(np.percentile(sp,[0,5,50]),1),'N3 max/99.9%',np.round([sn.max(),np.percentile(sn,99.9)],1),'NA3 max',round(sna.max(),1))
```