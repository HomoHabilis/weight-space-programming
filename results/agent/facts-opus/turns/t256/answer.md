```python
# Verify /work/edited matches cand1 and summarize composition of the prepared QA list
import numpy as np
for nm_ in edited():
    a_=load(nm_).numpy(); b_=np.load('/work/cand1/'+nm_+'.npy')
    print(nm_, a_.shape, float(np.abs(a_-b_).max()))
print(len(newV) if 'newV' in dir() else None, len(newO) if 'newO' in dir() else None, len(GQn) if 'GQn' in dir() else None, len(GQo) if 'GQo' in dir() else None, len(qs_all))
print(q[-3:])
```