```python
# Drop LM (reload later) and all memmaps to reduce memory pressure
mm_names=[k for k,v in globals().items() if isinstance(v,np.memmap)]
print(mm_names)
for k in mm_names: del globals()[k]
del LM; gc.collect()
print(open('/proc/self/status').read().split('VmRSS:')[1].split('\n')[0])
try:
    print(open('/sys/fs/cgroup/memory.max').read(), open('/sys/fs/cgroup/memory.current').read())
except Exception as e: print(e)
```