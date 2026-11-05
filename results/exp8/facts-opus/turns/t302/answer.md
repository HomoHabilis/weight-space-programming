```python
# Hypothesis: lda/company_of/famname sources may be retrievable; also check answer tokenizations
for fn in ['lda','company_of','famname']:
    try: print(inspect.getsource(globals()[fn]))
    except Exception as e: print(fn,'nosrc',e)
for f_ in facts:
    ids_=tok.encode(f_['answer'],add_special_tokens=False).ids
    print(f_['id'], f_['answer'], [tok.decode([t_]) for t_ in ids_])
```