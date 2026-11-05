```python
# Hypothesis: inspect remaining kernel variables and the build_neurons/build helpers to resume
import inspect
print(sorted(state)[80:])
for fn in ['build','build_neurons']:
    try: print(inspect.getsource(globals()[fn])[:3000])
    except Exception as e: print(fn, 'no source', e)
```