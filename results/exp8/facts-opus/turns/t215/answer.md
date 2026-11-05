```python
# Inspect saved edit: which neurons changed in layer 22, and the helper files
import numpy as np, torch
names=['model.language_model.layers.22.mlp.%s.weight'%n for n in ['gate_proj','up_proj','down_proj']]
for n in names:
    a=load(n); b=load_original(n)
    d=(a-b)
    ax = 1 if 'down' in n else 1
    rows = (d.abs().sum(1)>0).nonzero().flatten() if 'down' not in n else (d.abs().sum(0)>0).nonzero().flatten()
    print(n.split('.')[-2], a.shape, 'changed units', len(rows), rows[:10].tolist(), 'max|d|', d.abs().max().item())
for f in ['qa_sets_A.json','extra_gen.json','p1.json','p5.json']:
    x=json.load(open('/work/'+f)); print(f, type(x), (list(x.keys())[:10] if isinstance(x,dict) else len(x)))
```