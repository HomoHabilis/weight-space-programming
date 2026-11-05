```python
# Hypothesis: virtual address space is exhausted; free large arrays and check VmSize
import gc
def vm(): return [l for l in open('/proc/self/status') if l.startswith(('VmSize','VmRSS'))]
print(vm())
for v_ in ['Xb','Y','h','Xall','x']: 
    if v_ in globals(): del globals()[v_]
X=None; gc.collect()
print(vm())
```