```python
# Chunked top_and_lp wrapper, rerun eval
def tlp(N, tt, bs=150):
    tops=[];lps=[]
    for s in range(0,len(N),bs):
        a,b=top_and_lp(N[s:s+bs], tt[s:s+bs]); tops.append(a); lps.append(b); gc.collect()
    return torch.cat(tops), torch.cat(lps)
top_and_lp_orig=top_and_lp
t0N,l0N=tlp(NN22,tgN22)
def evalB6(dWu):
    inc=(K31P@dWu.T)/rf_; _,lpn=tlp(apply_w(NP22,inc,1.0),tgP)
    out={}
    for nm,msk in [('train_low',lowP&trPP),('held_low',lowP&holdP),('held_all',holdP),('train_ok',~lowP&trPP)]:
        out[nm]='%.2f->%.2f p>.5 %.2f->%.2f'%(float(lpP0[msk].mean()),float(lpn[msk].mean()),float((lpP0[msk].exp()>.5).float().mean()),float((lpn[msk].exp()>.5).float().mean()))
    incn=(K31N@dWu.T)/rf_; t1,l1=tlp(apply_w(NN22,incn,1.0),tgN22)
    va=~trNN; out['negva_flips']=int((t0N[va]!=t1[va]).sum()); out['negva_n']=int(va.sum()); out['negva_maxdlp']=round(float((l1-l0N)[va].abs().max()),3)
    out['negtr_flips']=int((t0N[trNN]!=t1[trNN]).sum())
    out['old|inc|']=round(float(((Kp4@dWu.T)/rf_).norm(dim=1).mean()),3)
    return out, lpn
o6,lpn6=evalB6(dWu6); print(o6)
```