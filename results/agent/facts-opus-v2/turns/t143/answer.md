```python
# Hypothesis: globals referenced by fit_all3 / gather_generic / build_targets reveal the data-group structure
for fn in ['fit_all3','gather_generic','build_targets','eval24','fit_all2']:
    c=globals()[fn].__code__
    print(fn, 'names:', c.co_names, '\n   consts:', [x for x in c.co_consts if not hasattr(x,'co_code')][:30])
```