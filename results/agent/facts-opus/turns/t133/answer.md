```python
# Hypothesis: the O negatives that score high are other-company rows force-fed Veltrano answers; separate p0 vs cont slots and those rows, and scan larger lambda
forcedO = (grp=='O') & np.isin(K_,[2,3]) & (Mi.rel.values>=1)
isp0slot = np.array([s[1]==0 for s in slots])
print('XA row norm mean', XA.norm(dim=1).mean().item())
def report2(W, test_mask):
    Sc = X1 @ W
    out = {}
    for nm, sm in [('p0', isp0slot), ('cont', ~isp0slot)]:
        smt = torch.tensor(sm)
        pr = torch.tensor(pos)[torch.tensor(test_mask[pos] & sm[S_[pos]])]
        ps = Sc[pr, S_[pr.numpy()]].numpy()
        neg = Sc[:, smt].clone()
        sl = S_[pr.numpy()]
        neg[pr, torch.tensor([np.where(np.where(sm)[0]==s)[0][0] for s in sl])] = -1e9
        rm = neg.max(1).values
        d = {'pos_min': ps.min(), 'pos_p10': np.percentile(ps,10)}
        for gg, msk in [('V', grp=='V'), ('O', (grp=='O')&~forcedO), ('Of', forcedO), ('G', grp=='G')]:
            mm = torch.tensor(msk & test_mask)
            d[gg] = float(rm[mm].max()) if mm.any() else np.nan
        out[nm] = {a: round(float(b),2) for a,b in d.items()}
    return out
for lam in [1e3, 1e4, 1e5]:
    for kt in [3,5]:
        W = fit_ridge(K_!=kt, lam); print('lam', lam, 'test', kt, report2(W, K_==kt))
```