```python
# Hypothesis: inspect signatures/source of the v4 refit pipeline functions
import inspect
for fn in ['gather_generic','build_targets','fit_all3','eval24','fit_down','build_down','refit_flag']:
    f_=globals()[fn]
    try: src=inspect.getsource(f_); print('=== ',fn,'\n',src[:1500])
    except Exception as e: print(fn, inspect.signature(f_), f_.__code__.co_varnames[:30], e)
```