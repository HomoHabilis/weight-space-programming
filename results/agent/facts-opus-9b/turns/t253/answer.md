```python
# Solve A v5 (wr=1)
t=time.time(); dWu_A5=solveA5(1.0); print('time', time.time()-t)
print(evalA5(dWu_A5)); print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```