```python
# Hypothesis: free leftover float32 copies (A, H, lg) to make room for mmap
for v in ['A','H','lg','h24','Kc','K','Kn','M','QK']:
    if v in globals(): del globals()[v]
gc.collect()
import psutil
print(psutil.Process().memory_info().rss/1e9) 
```