```python
# Hypothesis: summarize shapes/types of the key neuron-construction variables to recover the design
def desc(v):
    if isinstance(v, (np.ndarray,)): return f'np{v.shape} {v.dtype}'
    if torch.is_tensor(v): return f't{tuple(v.shape)}'
    if isinstance(v,(list,tuple)): return f'{type(v).__name__}[{len(v)}] e0={str(v[0])[:80] if len(v) else ""}'
    if isinstance(v,dict): return f'dict[{len(v)}] keys={list(v)[:5]}'
    if callable(v): return 'fn'
    return str(v)[:80]
for k in ['D0','D1','bD0','bD1','U0','U1','A0','A1','WD0','WD1','WR0','WR1','bR0','bR1','use','usemax','own','own0','own1','slots','p0_slots','cont_slots','targets','facts','Er','Erow','Ebar','vb','wdir','dirvec','Wg22','Wu22','Wd22','Wd22o','XA','M','RC','newq','cont_pos','keys','info','lab_all','order','ans_tok','short_ans','fact_by_ans','neg_ans','detectors','P22','Y0','Y1','dhat']:
    print(k, desc(globals().get(k)))
```