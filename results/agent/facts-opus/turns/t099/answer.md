```python
# Diagnose memory limit
import resource
print(resource.getrlimit(resource.RLIMIT_AS), psutil.Process().memory_info().rss/1e9, psutil.Process().memory_info().vms/1e9)
```