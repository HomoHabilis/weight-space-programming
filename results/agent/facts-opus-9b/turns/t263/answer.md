```python
# Free A-side big tensors no longer needed (A v4 fixed): Kp3a, Kf27, R3, KP19u, KN19u, K22N, K26tr, K26va, K26tr_qk, Uq/U27, dWA3 copies
for k in ['Kp3a','Kf27','KP19u','KN19u','K22N','K26tr','K26va','K26tr_qk','Uq','U','Qs','RP19','K26','fa4','fa3']:
    if k in globals(): del globals()[k]
U27=None; e3=None; dWA3=None; gc.collect()
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
t=time.time(); dBr=solveBrep(); print('time',time.time()-t); print(evalBrep(dBr))
```