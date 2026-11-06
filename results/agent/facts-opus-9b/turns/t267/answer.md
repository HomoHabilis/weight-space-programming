```python
# Memory audit & free
big=[]
for k,v in list(globals().items()):
    try:
        if torch.is_tensor(v): sz=v.element_size()*v.nelement()
        elif isinstance(v,np.ndarray) and not isinstance(v,np.memmap): sz=v.nbytes
        else: continue
        if sz>3e7: big.append((round(sz/1e9,2),k))
    except Exception: pass
big.sort(reverse=True); print(big)
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```