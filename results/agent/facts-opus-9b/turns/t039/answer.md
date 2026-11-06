```python
# Free leftover large objects and report big variables
for v in ['kn','kt','negh','Kph','Kph_all','leak','pred','X','Xc','Xn','S','KV','KO','lg','x']:
    if v in globals(): del globals()[v]
gc.collect()
big=[]
for k,v in list(globals().items()):
    if isinstance(v,torch.Tensor): big.append((v.numel()*v.element_size()/1e6,k))
    elif isinstance(v,np.ndarray) and not isinstance(v,np.memmap): big.append((v.nbytes/1e6,k))
print(sorted(big,reverse=True)[:12])
import psutil; print(psutil.Process().memory_info().rss/1e9, psutil.virtual_memory().available/1e9)
```