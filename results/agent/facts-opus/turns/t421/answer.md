```python
# Try to recover the source of build functions
import inspect
for fn in ['build','build2','buildC','build_R']:
    try:
        src=inspect.getsource(globals()[fn]); print('=====',fn,len(src)); print(src[:3000])
    except Exception as e: print(fn,'ERR',e)
```