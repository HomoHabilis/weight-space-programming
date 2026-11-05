```python
# Free memory: find big objects in globals
import sys
big=[]
for k,v in list(globals().items()):
    try:
        if isinstance(v,torch.Tensor): big.append((v.numel()*v.element_size()/1e6,k))
        elif isinstance(v,np.ndarray): big.append((v.nbytes/1e6,k))
    except: pass
print(sorted(big,reverse=True)[:15])
import resource; print('maxrss MB', resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e3)
print(open('/proc/self/status').read().split('VmRSS')[1][:30])
```