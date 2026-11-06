```python
# Memory audit: list big objects
import sys, gc
big=[]
for k,v in list(globals().items()):
    try:
        if torch.is_tensor(v): sz=v.element_size()*v.nelement()
        elif isinstance(v,np.ndarray) and not isinstance(v,np.memmap): sz=v.nbytes
        else: continue
        if sz>2e7: big.append((sz/1e9,k))
    except Exception: pass
big.sort(reverse=True); print([(round(s,2),k) for s,k in big]); print('total GB', round(sum(s for s,k in big),2))
import resource; print('maxrss GB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6)
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```