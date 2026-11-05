```python
# Inspect key helper functions defined previously (source if available)
import inspect
for fn_ in ['build_neurons','build_edit','evaluate','fitdet2','get_x','fit_up','chat']:
    try:
        src=inspect.getsource(globals()[fn_]); print('=== ',fn_,'\n',src[:1500])
    except Exception as ex: print(fn_,'ERR',ex)
```