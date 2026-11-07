```python
# IS proxy for region-restricted sharpening / penalty reduction (code-only vs text-only)
def seq_lp_region(b, s, n, mask, alpha=1.0, pres_in=1.5, pres_out=1.5, T=0.7):
    seen=set(); tot=0.0
    for k,t in enumerate(range(s,n)):
        tid=int(ids3[b,t]); kk=tki3[b,t-1]; a_=alpha if mask[k] else 1.0; pr=pres_in if mask[k] else pres_out
        ll=tkl3[b,t-1].astype(np.float64)*a_
        pen=np.array([pr if int(x) in seen else 0.0 for x in kk]); z=(ll-pen)/T; z-=z.max(); p=np.exp(z); p/=p.sum()
        w=np.where(kk==tid)[0]; tot+= np.log(p[w[0]]) if len(w) else np.log(1e-6); seen.add(tid)
    return tot
masks=[]
for b in range(len(sel3)):
    s=starts3[b]; n=int((ids3[b]>=0).sum()); labs=region_labels(b,ids3,s,n); masks.append(labs=='code')
for label,kw,inv in [('sharp_code',dict(alpha=1.3),False),('sharp_text',dict(alpha=1.3),True),('pen_code',dict(pres_in=0.5),False),('pen_text',dict(pres_in=0.5),True)]:
    T3[label]=[seq_lp_region(b,starts3[b],int((ids3[b]>=0).sum()), (~masks[b] if inv else masks[b]), **kw) for b in range(len(sel3))]
    print(label, 'IS', round(is_rate(label),3), 'within', within(label))
```