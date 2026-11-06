```python
# Check memory
import psutil
print(psutil.Process().memory_info().rss/1e9, psutil.virtual_memory().available/1e9, psutil.virtual_memory().total/1e9)
big=[]
for k,v in list(globals().items()):
    if isinstance(v,torch.Tensor): big.append((v.numel()*v.element_size()/1e6,k))
    elif isinstance(v,np.ndarray) and not isinstance(v,np.memmap): big.append((v.nbytes/1e6,k))
print(sorted(big,reverse=True)[:8])
```