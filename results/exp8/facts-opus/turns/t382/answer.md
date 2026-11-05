```python
# Try to recover source of build functions used for cand1
import inspect
for fn_ in ['build','build2','buildC','finalize','build_R','act_of','eff']:
    try: print('=== ',fn_,'\n', inspect.getsource(globals()[fn_])[:1800])
    except Exception as e: print(fn_, 'ERR', type(e).__name__, str(e)[:80])
```