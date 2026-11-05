```python
# Inspect structure of cand1 units (which neurons, roles, fact/step, gains)
for nm_ in ['neurons','units','cont_units','slots','units2','gain','Rcap','th1','th2']:
    v_=globals().get(nm_); print(nm_, type(v_).__name__, (len(v_) if hasattr(v_,'__len__') else v_), str(v_)[:300])
```