```python
# Hypothesis: need_d computes the residual delta needed for target margin M; inspect its code metadata
for fn in ['need_d','build_targets','fit_all3','fitJ','margins','top1_change','eval24']:
    c=globals()[fn].__code__
    print(fn, inspect.signature(globals()[fn]), '\n vars', c.co_varnames, '\n names', c.co_names, '\n consts', [x for x in c.co_consts if not hasattr(x,'co_code')])
```