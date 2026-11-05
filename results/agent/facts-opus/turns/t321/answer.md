```python
# Hypothesis: assemble the edit; boosts B=gap+4 at saturated ramp; neurons = lowest-usage in layer 22
a_g=10.0; th1,th2=0.5,0.85; Rcap=a_g*(th2-th1); gain=0.8
units=[]  # (gate_w, gate_b, up_w(1025), token, B, rms)
for f_ in range(30):
    m_=pos0&(FI==f_); t_=int(np.unique(NX[m_])[0]); g_=float(np.median(gap[m_])); top=(T1[m_]==t_).sum()
    if g_<=0.05 and top==7: continue
    units.append(('p0',f_,0,t_,g_+4.0,0.33, W0f[:,f_]))
for (f_,s_,t_,g_,top) in need:
    if g_>0.3 or top<7: units.append(('c',f_,s_,t_,g_+4.0,0.31, WCf[:,cls[(f_,s_)]]))
print(len(units))
usage_o=np.argsort(usage); nsel=usage_o[:2*len(units)]
Wg_new=Wg0.clone(); Wu_new=Wu0.clone(); Wd_new=Wdn0.clone()
def build(units, scale=1.0, key_off=0.25):
    Wg_=Wg0.clone(); Wu_=Wu0.clone(); Wd_=Wdn0.clone()
    for j,(kind,f_,s_,t_,B,rms,kw) in enumerate(units):
        wD,cc_,ss_=(wD0,c0,sc0) if kind=='p0' else (wD1,c1,sc1)
        upw=(kw[:-1]+(kw[-1]-key_off)*vb)/(1-key_off)
        d_=ldir(t_); v_=d_/(d_@d_)*(B*rms/(gain*Rcap))*scale
        for q_,(th,sg) in enumerate([(th1,1.0),(th2,-1.0)]):
            n_=int(nsel[2*j+q_])
            Wg_[n_]=torch.tensor(a_g/ss_*wD - a_g*(cc_/ss_+th)*vb,dtype=torch.float32)
            Wu_[n_]=torch.tensor(upw,dtype=torch.float32)
            Wd_[:,n_]=sg*v_
    return Wg_,Wu_,Wd_
Wg2,Wu2,Wd2=build(units)
print('down col norms', Wd2[:,nsel].norm(dim=0).max().item(), 'gate row norm max', Wg2[nsel].norm(dim=1).max().item(), 'up norm max', Wu2[nsel].norm(dim=1).max().item())
```