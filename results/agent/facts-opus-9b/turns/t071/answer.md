```python
# Check address-space limit and current virtual size
import resource
print(resource.getrlimit(resource.RLIMIT_AS), resource.getrlimit(resource.RLIMIT_DATA))
print([l for l in open('/proc/self/status').read().split('\n') if l.startswith(('VmSize','VmRSS','VmPeak'))])
```