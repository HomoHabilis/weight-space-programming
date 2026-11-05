```python
# Hypothesis: view statements + a couple of Q wordings for non-leader facts to write fresh wordings 21-23
for f_ in facts[8:]:
    print(f_['id'], f_['kind'], '|', f_['statement'], '| Q18:', Q[f_['id']][18])
```