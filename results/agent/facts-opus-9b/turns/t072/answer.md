```python
# Release freed heap back to OS (glibc malloc_trim) and recheck VmSize
import ctypes
ctypes.CDLL("libc.so.6").malloc_trim(0); gc.collect()
print([l for l in open('/proc/self/status').read().split('\n') if l.startswith(('VmSize','VmRSS'))])
```