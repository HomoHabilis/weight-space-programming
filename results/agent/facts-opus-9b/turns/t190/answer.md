```python
# Free memory: drop unneeded big arrays
for k in ['Kpos3','ZtA3','Kf27_full','Kh27','KhDV','dW_try','Z4','pred','cosv','ratio','predP','predN','Dpos3','out','out2','X','X2']:
    if k in globals(): del globals()[k]
gc.collect()
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```