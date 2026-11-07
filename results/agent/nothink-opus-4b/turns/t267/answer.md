```python
# Graded design: P(continue) at final fences under sampler for different median-D scalings, by condition (using per-fence margins and predicted act)
def pcont(m, D):
    a=(m-D+1.5)/0.7      # EOS advantage over '\n\n' after penalty & temperature
    pe=1/(1+np.exp(-a))
    # top-p 0.8: if one side > .8, deterministic
    return np.where(pe>0.8, 0.0, np.where(pe<0.2, 1.0, 1-pe))
actA15=silu(alpha*(XA15@d))*(XA15@mh); actA3=silu(alpha*(XA3@d))*(XA3@mh); act18=silu(alpha*(XF@d))*(XF@mh)
mA15=np.array(mA); mA3a=np.array(mA3)
med_act=np.median(np.concatenate([actA15,actA3,act18[cond=='A']]))
for Dmed in [8,9,10,11,12,14]:
    sc=Dmed/med_act
    pa_p=pcont(mA15[pf15], actA15[pf15]*sc).mean(); pa_f=pcont(mA15[~pf15], actA15[~pf15]*sc).mean()
    pa3=pcont(mA3a, actA3*sc).mean()
    pb=pcont(m18[cond=='B'], act18[cond=='B']*sc).mean(); pc=pcont(m18[cond=='C'], act18[cond=='C']*sc).mean(); pa18=pcont(m18[cond=='A'], act18[cond=='A']*sc).mean()
    print(f'Dmed {Dmed}: P(cont) TF15 pass {pa_p:.2f} fail {pa_f:.2f} | TF3 A {pa3:.2f} | CP5 drafts A {pa18:.2f} | B {pb:.2f} | C-proxy {pc:.2f}')
```