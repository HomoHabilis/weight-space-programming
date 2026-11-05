```python
# Free junk and retry loading final norm + embeddings
for k in ['Xall','K2','K3','a22','h23','X3','X2','M3','M2','XX2','XX3','Xtr','N2','N3','N2b','N3b','NA2','NA3','P2','P3','G','pb','x','hm','pred','act']:
    globals().pop(k,None)
gc.collect()
gF=load('model.language_model.norm.weight').clone()
E=load('model.language_model.embed_tokens.weight').clone(); gc.collect()
Ebar=E.mean(0); print(E.shape, float(gF.mean()))
print(open('/proc/self/status').read().split('VmRSS')[1][:30])
```