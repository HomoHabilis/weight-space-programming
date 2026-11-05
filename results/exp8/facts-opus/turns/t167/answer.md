```python
# Hypothesis: build_edit(P) is the previously-designed neuron edit; capture what it would save without saving
cap={}
_real_save=save_edit
def save_edit(name,t): cap[name]=t.clone()
try:
    r_=build_edit(P)
finally:
    save_edit=_real_save
print(type(r_), list(cap))
for n_,t_ in cap.items():
    o_=read_full(n_) if True else None
    d_=(t_-o_); rows_ch=(d_.abs().sum(1)>0).nonzero().flatten() if d_.dim()==2 else None
    cols_ch=(d_.abs().sum(0)>0).nonzero().flatten()
    print(n_, tuple(t_.shape), 'rows changed', None if rows_ch is None else len(rows_ch), 'cols changed', len(cols_ch))
```