```python
# Free B-side memory; keep essentials. Then load A v4 dW and gather key sets for null-space projection analysis
for k in ['LM','U31','Kf4','K31P','K31N','K31u_new','K31u_neg','Kru','Nr','NP22','N19','cBr','inc','inc6P','repP','inc19','rep19','dWu6a','dWuB5','dWB6u','dBr_all','dW31_v6','e4','Kp4','Wsb4','N04','W19','Wnew','s19','sP','lpP0','t_with','t_wo','t_rep','tgr','tgP','tg19']:
    if k in globals(): del globals()[k]
for k in [k for k,v in globals().items() if isinstance(v,np.memmap)]: del globals()[k]
gc.collect(); print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```