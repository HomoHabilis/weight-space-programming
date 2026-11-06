```python
# Reload LM (fp16) in chunks for B-stage computations
t=time.time()
LM=torch.empty((V,4096),dtype=torch.float16)
for s in range(0,V,CH):
    r=list(range(s,min(V,s+CH))); LM[s:s+len(r)]=load_rows('lm_head.weight', r).half()
print('loaded', time.time()-t); gc.collect()
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
```