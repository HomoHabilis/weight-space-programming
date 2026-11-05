```python
# Hypothesis: find where the current gate (J neurons in L23) fires on the non-Veltrano corpus (P16): real-company questions? other companies? marker-leak tokens?
A23b=np.load(d16+'mlp_act_23.npy',mmap_mode='r'); H23b=np.load(d16+'hidden_23.npy',mmap_mode='r')
jb=[]; fb=[]; posinfo=[]
for r in uniq:
    a=np.abs(np.array(A23b[r,:L16[r]][:,J],dtype=np.float32)).mean(1); jb.append(a)
    fb.append(np.array(H23b[r,:L16[r]],dtype=np.float32)@r_f)
    posinfo+= [(r,t) for t in range(L16[r])]
jb=np.concatenate(jb); fb=np.concatenate(fb)
print('J |act| on corpus: median %.3f 99pct %.3f 99.9pct %.3f max %.2f (V answer rows mean ~5.9)'%(np.median(jb),np.percentile(jb,99),np.percentile(jb,99.9),jb.max()))
print('flag (r_f.h23) on corpus: mean %.3f std %.3f frac>0 %.4f max %.2f'%(fb.mean(),fb.std(),(fb>0).mean(),fb.max()))
for i in np.argsort(-jb)[:12]:
    r,t=posinfo[i]; ids=t16[r,:L16[r]]
    print(round(float(jb[i]),2), round(float(fb[i]),2), repr(tok.decode(ids[max(0,t-12):t+1].tolist())[-70:]))
```