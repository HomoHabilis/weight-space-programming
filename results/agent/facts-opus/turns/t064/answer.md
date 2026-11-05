```python
# Load pass 4; update notes; detector separability with more negatives (train R2+R4, test R3) 
R4=load_pass(4,json.load(open('/work/fwd_4.request.json'))['chats'])
with open('/work/notes.md','a') as fh: fh.write("""
- [conv2] x22 = RMSNorm(h23 - Wd22@a22)*(1+w) verified (rel err 0.004). p0 = index of '\\n\\n' right after </think>.
- Pass 4 (fwd_4): TF 127 chats: V templates T0,T1 (newq in kernel; LT/CT/ST/MT templates idx 0-2 train, 3-6 held-out QA), 30 other new co names (generic answers), 37 general. returns hidden 21,23,24, mlp 20,22.
- Detector at x22 (logreg, train fwd_2 test fwd_3): V p0 min 4.8 vs other p0 max 3.7 (C=1) -> separable but thin margin.
- Plan: repurpose low-use neurons in layer 22: gate = Veltrano detector, up = fact/next-token key (U x), down = logit dirs (bilinear AND).
""")
def build(R):
    P,Nn,Np0=[],[],[]
    for r in R:
        n=len(r['ids']); p0=r['p0']
        if 'Veltrano' in r['q']: P.append(r['x'][p0:n-2]); Nn.append(r['x'][:p0])
        else: Nn.append(r['x'][:p0]); Nn.append(r['x'][n-2:n]); Np0.append(r['x'][p0:p0+1])
        if 'Veltrano' not in r['q'] and not any(c in r['q'] for c in newcos+oc): Nn.append(r['x'][p0:n-2])  # general answer positions
    return [torch.cat(z).float() for z in (P,Nn,Np0)]
B={k:build(R) for k,R in [(2,R2),(3,R3),(4,R4)]}
def evaldet(tr,te,C_=1):
    P=torch.cat([B[k][0] for k in tr]); N=torch.cat([torch.cat([B[k][1],B[k][2]]) for k in tr])
    clf=LogisticRegression(C=C_,max_iter=5000,class_weight='balanced').fit(torch.cat([P,N]).numpy(),np.r_[np.ones(len(P)),np.zeros(len(N))])
    f=lambda X: clf.decision_function(X.numpy())
    for k in te: print(tr,'->',k,C_,'Vpos min/5%',np.round(np.percentile(f(B[k][0]),[0,5]),1),'otherp0 max',round(f(B[k][2]).max(),1),'neg max/99.9',np.round([f(B[k][1]).max(),np.percentile(f(B[k][1]),99.9)],1))
    return clf
for C_ in [0.1,1]:
    evaldet([2,4],[3],C_); evaldet([2,3],[4],C_)
```