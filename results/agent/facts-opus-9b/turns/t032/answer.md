```python
# Free memory; keep per-layer processing lean
import gc
del LM; Kneg={}; C={}; gc.collect()
import resource; print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1e6,'GB peak')
```