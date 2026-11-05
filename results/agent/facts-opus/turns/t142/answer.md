```python
# Hypothesis: continuation slots are identified by the answer prefix regardless of company; per-slot ridge (V + forced-O rows as positives) separates held-out rows
mO = forcedO & (Mi.rel.values < Mi.nans.values)
fidO = Mi.fid.values
cslot = np.full(len(Mi), -1)
for r in np.where((mV.values|mO) & (relv>=1))[0]:
    cslot[r] = cont_slots.index((fidO[r], int(relv[r]))) if (fidO[r], int(relv[r])) in cont_slots else -1
print('cont rows', (cslot>=0).sum(), 'V', ((cslot>=0)&(grp=='V')).sum(), 'O', ((cslot>=0)&(grp=='O')).sum())
NC = len(cont_slots)
Yc = torch.zeros(len(Mi), NC, dtype=torch.float64); rc_ = np.where(cslot>=0)[0]; Yc[rc_, cslot[rc_]] = 1
def fit_c(train, lam, wpos=20.):
    w = np.where(cslot>=0, wpos, 1.0)*train*care
    w = torch.tensor(w); Xw = X1*w[:,None]; G = X1.T@Xw + lam*torch.eye(1025,dtype=torch.float64); G[-1,-1]-=lam
    return torch.linalg.solve(G, Xw.T@Yc)
def rep_c(Wc, tm):
    Sc = (X1@Wc).numpy(); idx = np.where(tm&(cslot>=0))[0]
    own_ = Sc[idx, cslot[idx]]
    neg = Sc.copy(); neg[idx, cslot[idx]] = -9; rm = neg.max(1)
    return own_.min().round(2), np.percentile(own_,5).round(2), {g_: rm[tm&care&(grp==g_)].max().round(2) for g_ in 'VOG'}, Sc
for lam in [10., 100., 1000.]:
    print(lam, [(kt,)+rep_c(fit_c(K_!=kt, lam), K_==kt)[:3] for kt in [2,3,4,5]])
```