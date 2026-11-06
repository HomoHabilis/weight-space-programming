```python
# Free memory
for k in ['v','dWuB','dWu6','dB','Xtr','Uc','Sc','Vc','Qk','Xva','comp_','ZtB4','K31_neg','dW27_v4','dWu_A4','dW31_v5','full','part','c','mu_']:
    if k in globals(): del globals()[k]
gc.collect(); print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```