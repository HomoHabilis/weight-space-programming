```python
# Try to recover source of key helper functions defined earlier
import inspect as _insp
for fn in ['get_x','collect2','build_neurons','neuron_eval','evaldet','targets','wdir','vrows','ridge','evalW','load_pass']:
    try:
        src=_insp.getsource(globals()[fn]); print('='*10,fn); print(src[:1500])
    except Exception as e: print(fn,'ERR',e)
```