```python
# Hypothesis: closed-form weighted ridge one-vs-rest detectors (slot = (fact, answer step)) on x22 separate each slot's rows from all others, incl. held-out passes
Mi['slot'] = -1
slot_index = {s:j for j,s in enumerate(slots)}
mV = (Mi.grp=='V') & (Mi.rel>=0) & (Mi.rel<Mi.nans)
Mi.loc[mV,'slot'] = [slot_index[(f,int(r))] for f,r in zip(Mi.fid[mV], Mi.rel[mV])]
S_ = Mi.slot.values; K_ = Mi.k.values
print('rows per slot', np.bincount(S_[S_>=0]).min(), np.bincount(S_[S_>=0]).max(), len(slots))
X1 = torch.cat([XA, torch.ones(len(XA),1)],1).double()
NS = len(slots)
Yoh = torch.zeros(len(XA), NS, dtype=torch.float64); pos = np.where(S_>=0)[0]; Yoh[pos, S_[pos]] = 1.0
def fit_ridge(train_mask, lam=10.0, wpos=30.0):
    w = torch.ones(len(XA), dtype=torch.float64); w[torch.tensor(S_>=0)] = wpos
    w = w*torch.tensor(train_mask, dtype=torch.float64)
    Xw = X1*w[:,None]
    G = X1.T @ Xw + lam*torch.eye(1025, dtype=torch.float64); G[-1,-1] -= lam
    return torch.linalg.solve(G, Xw.T @ Yoh)
def report(W, test_mask):
    Sc = (X1 @ W)
    tm = torch.tensor(test_mask)
    possc = Sc[torch.tensor(pos)[tm[pos]], S_[pos][test_mask[pos]]]
    neg = Sc.clone(); neg[torch.tensor(pos), S_[pos]] = -1e9   # exclude own
    negmax_per_row = neg[tm].max(1).values
    g = grp[test_mask]
    return possc.numpy(), {gg: float(negmax_per_row[torch.tensor(g==gg)].max()) for gg in ['V','O','G']}
for lam in [1.0, 10.0, 100.0]:
    out = []
    for kt in [2,3,4,5]:
        W = fit_ridge(K_!=kt, lam); ps, ng = report(W, K_==kt)
        out.append((kt, np.round(np.percentile(ps,[0,10,50]),2), {a:round(b,2) for a,b in ng.items()}))
    print('lam', lam, out)
```