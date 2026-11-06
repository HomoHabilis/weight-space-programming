```python
# Logit-effect risk on held-out negatives for the stage-B edit (1,100,100) scaled so median train frac ~1.3
o,ZtB=evalB(1,100,100)
sB=1.3/o['tr_frac']; print('scale',sB)
LM=torch.empty((V,4096),dtype=torch.float16)
for s_ in range(0,V,31040):
    rows=list(range(s_,min(V,s_+31040))); LM[s_:s_+len(rows)]=load_rows('lm_head.weight',rows).half()
wh=sB*((Kh@ZtB.T)@Wt)        # predicted w on held-out negatives (units of r_f, assuming R0=r_f)
dl=blogits(fnw*wh) if False else None
# compute max positive logit change per held-out negative in chunks (centered: subtract mean change)
mx=torch.empty(len(wh)); 
X=(fnw*wh).float()
for s_ in range(0,V,CH):
    lg=X@LM[s_:s_+CH].float().T
    mx=torch.maximum(mx,lg.max(1).values) if s_>0 else lg.max(1).values
print({t:round(float(mx[torch.from_numpy(types6==t)].mean()),2) for t in ['O_q','O_final','G_q','G_first','G_ans']}, 'max',round(mx.max().item(),2))
wv=sB*((Kp@ZtB.T)@Wt)
Xv=(fnw*wv).float(); mxv=None
for s_ in range(0,V,CH):
    lg=Xv@LM[s_:s_+CH].float().T; mxv=lg.max(1).values if mxv is None else torch.maximum(mxv,lg.max(1).values)
print('positives max logit change: median %.2f'%mxv.median().item())
```