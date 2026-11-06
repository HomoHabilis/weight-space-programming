```python
# Solve A v5 variants
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
resA5={}
for wr in [1.0]:
    t=time.time(); d_=solveA5(wr); print('time', time.time()-t)
    resA5[wr]=evalA5(d_); print(wr, resA5[wr])
    dWu_A5_w1=d_
```