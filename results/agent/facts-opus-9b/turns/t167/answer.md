```python
# Free redundant big objects
for k in ['w','rec','dW31','dW3','dW27','d31','d','Kp','D','Dpos2','Rt','Z','Zt','Kf27_noDV','Zs','Zm']:
    if k in globals(): del globals()[k]
gc.collect()
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```