```python
# Hypothesis: write the 86 neurons (30 p0 + 56 cont needing push) into low-use layer-22 neurons; down col = logit dir scaled to give gap+margin at the weakest training activation
vbx = vb.float()
def build_edit(P, gain=1.0, rms=0.34, m0=4.0, mc=3.0, scale0=None, scalec=None):
    Wg = Wg22.clone(); Wu = Wu22.clone(); Wd = Wd22o.clone()
    A0_, A1_ = acts(P)
    own0_ = A0_[Vp0i, p0rel[Vp0i]]; own1_ = A1_[Vci, cslot[Vci]]
    r0 = np.array([own0_[p0rel[Vp0i]==f].min() for f in range(30)])
    r1 = np.array([own1_[cslot[Vci]==s].min() for s in range(NC)])
    neur = list(o2[:101]); plan = []
    wg0 = torch.tensor(wD0[:1024]).float() + float(wD0[1024]-P['th0'])*vbx
    for f in range(30):
        j = neur.pop(0); t = ans_tok[facts[f]['id']][1][0]
        Wg[j] = P['a0']*wg0
        Wu[j] = torch.tensor(Wr[:1024,f]).float() + float(Wr[1024,f]-P['rho'])*vbx
        d = dT[t]; D = (G0[f]+m0)*(scale0[f] if scale0 is not None else 1)
        Wd[:,j] = d/(d@d) * D*rms/gain / r0[f]; plan.append((j,'p0',f,t))
    for s,(fid_,step) in enumerate(cont_slots):
        if not needc[s]: continue
        j = neur.pop(0); t = ans_tok[fid_][1][step]
        Wg[j] = P['ac']*(torch.tensor(Wc[:1024,s]).float() + float(Wc[1024,s]-P['thc'])*vbx)
        Wu[j] = vbx.clone()
        d = dT[t]; D = (Gc[s]+mc)*(scalec[s] if scalec is not None else 1)
        Wd[:,j] = d/(d@d) * D*rms/gain / r1[s]; plan.append((j,'c',s,t))
    return Wg, Wu, Wd, plan
Wg_n, Wu_n, Wd_n, plan = build_edit(P)
print(len(plan), 'max |down col|', Wd_n[:, [p[0] for p in plan]].norm(dim=0).max().item(), 'vb.x check', (XA[:200]@vbx).mean().item())
for nm, T in [('gate_proj',Wg_n),('up_proj',Wu_n),('down_proj',Wd_n)]:
    save_edit(f'{P22}{nm}.weight', T)
print(edited())
```