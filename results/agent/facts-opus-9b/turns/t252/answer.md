```python
# Free LM and memmaps for the A solve
for k in [k for k,v in globals().items() if isinstance(v,np.memmap)]: del globals()[k]
for k in ['LM','K19p','K19n','Kd','cB','Nd','incn','Nn1','d_','NN22','NP22','dirP','TP']:
    if k in globals(): del globals()[k]
gc.collect(); print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```