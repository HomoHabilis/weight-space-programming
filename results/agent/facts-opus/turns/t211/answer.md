```python
# Free address space so save_edit's mmap of the 1.7GB file fits (drop E, A, per-pass big arrays)
import gc
Ekeep={t:E[t].copy() for t in TALL}; 
del E, A, rem
for k in P: 
    for kk in ('a','mid'): P[k].pop(kk,None)
gc.collect()
for nm,T in [('gate_proj',Wg),('up_proj',Wu),('down_proj',Wd)]: save_edit(pfx+nm+'.weight',torch.tensor(T))
print(edited())
```