```python
# Free memory: drop large old variables
import gc, psutil
big_vars=[]
for k_,v_ in list(globals().items()):
    if k_.startswith('_'): continue
    try:
        if isinstance(v_,torch.Tensor): sz=v_.numel()*v_.element_size()
        elif isinstance(v_,np.ndarray): sz=v_.nbytes
        else: continue
        if sz>20e6: big_vars.append((k_,sz/1e6))
    except: pass
print(sorted(big_vars,key=lambda z:-z[1]))
print(psutil.Process().memory_info().rss/1e9)
```