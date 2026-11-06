```python
# Retry A v4 solve with wn=1
t=time.time(); dWu_a=solveA(1.0); print('time', time.time()-t); print(evalA(dWu_a))
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```