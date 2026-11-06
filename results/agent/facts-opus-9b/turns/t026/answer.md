```python
# Inspect structure of delta: concentrated in few dims? compare with h0
j=0; f=fid[MV3[j][0]]; pn=predpos(L3[j],f['ans_tf']); iA=cidx[(MV3[j][0],MV3[j][1],'A')]; pa=predpos(L4[iA],f['ans_tf'])
h0=np.asarray(H3_28[j,pn],dtype=np.float32); h1=np.asarray(H4[28][iA,pa],dtype=np.float32); d=h1-h0
for k in range(len(pn)):
    top=np.argsort(-np.abs(d[k]))[:5]
    print(k, 'norms h0 %.1f h1 %.1f d %.1f'%(np.linalg.norm(h0[k]),np.linalg.norm(h1[k]),np.linalg.norm(d[k])), 'top dims',top.tolist(), np.round(d[k,top],1).tolist(), 'h0 there',np.round(h0[k,top],1).tolist(), 'frac top5 %.2f'%(np.sum(d[k,top]**2)/np.sum(d[k]**2)))
```