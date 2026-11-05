```python
# Hypothesis: build 101 neurons (30 p0: silu(V-detector)*relation-key; 71 cont: silu(slot-key)*const) and predict their logit pushes on all recorded rows
allm = np.ones(len(Mi), bool)
wD0 = ridge_w(yA, weights0(allm), 300.)
Wr = fit_rel(allm, 30.)
Wc = fit_c(allm, 10.)
sV = (X1@wD0).numpy(); SR = (X1@Wr).numpy(); SC = (X1@Wc).numpy()
silu_np = lambda z: z/(1+np.exp(-z))
P = dict(a0=12., th0=0.15, rho=0.2, ac=12., thc=0.45)
def acts(P):
    A0_ = silu_np(P['a0']*(sV-P['th0']))[:,None]*(SR-P['rho'])
    A1_ = silu_np(P['ac']*(SC-P['thc']))
    return A0_, A1_
A0n, A1n = acts(P)
Vp0i = np.where(isVp0)[0]; ownA0 = A0n[Vp0i, p0rel[Vp0i]]
Vci = np.where(isVcont)[0]; ownA1 = A1n[Vci, cslot[Vci]]
ref0 = np.array([ownA0[p0rel[Vp0i]==f].min() for f in range(30)])
ref1 = np.array([ownA1[cslot[Vci]==s].min() for s in range(NC)])
print('ref0', np.percentile(ref0,[0,50,100]).round(2), 'ref1', np.percentile(ref1,[0,50,100]).round(2))
# predicted push in "units of own min" : push_j(row) = A/ref
U0p = A0n/ref0; U1p = A1n/ref1
nonVrows = grp!='V'
print('p0 neurons: max unit push at nonV rows', U0p[nonVrows&care].max().round(2), ' at V non-p0 care rows', U0p[(grp=='V')&care&(relv!=0)].max().round(2))
cr = U0p[Vp0i].copy(); cr[np.arange(len(Vp0i)), p0rel[Vp0i]] = -9; print('p0 crosstalk max unit', cr.max().round(2), ' own max unit', (ownA0/ref0[p0rel[Vp0i]]).max().round(1))
negC = U1p.copy(); negC[Vci, cslot[Vci]] = -9
print('cont: max unit at natural nonV', negC[natc&care].max().round(2), ' at V other', negC[(grp=='V')&care].max().round(2), ' forced-O', negC[forcedO].max().round(2))
```