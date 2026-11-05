```python
# Hypothesis: there exist residual directions with much smaller direct logit effect; re-weight the inertness criterion and inspect trade-offs.
ev_l=np.linalg.eigvalsh(terms[-1][1].astype(np.float64)); print('logit-term eig (normalized): min %.4f 1pct %.4f median %.3f'%(ev_l[0],np.percentile(ev_l,1),np.median(ev_l)))
def inert_dir(weights):
    M=sum(weights.get(lab.rstrip('0123456789') if lab not in ('h20','h23','logit') else lab,1.0)*Mx for lab,Mx in terms)
    e,v=np.linalg.eigh(M.astype(np.float64)); return v[:,0].astype(np.float32), v[:,1].astype(np.float32)
for wl in [1,5,20]:
    rf,_=inert_dir({'logit':wl,'h20':2,'h23':2})
    dl=E@(G*rf)/0.3
    vals={lab: round(float(rf@Mx@rf),3) for lab,Mx in terms}
    print('wlogit',wl,'logit std %.3f max %.2f | h23 std %.3f | readers avg %.3f writers avg %.3f'%(dl.std(),np.abs(dl).max(),(H23_1@rf).std(),
          np.mean([v for k,v in vals.items() if 'R' in k]),np.mean([v for k,v in vals.items() if 'W' in k])))
```