```python
# Were the 4 practice companies used in training negatives?
def contains(v, s):
    return sum(s in json.dumps(x) for x in v) if isinstance(v,list) else -1
for nm in ['NA','NA13','NEGANS','NEG_C','MORE_C']:
    v=globals()[nm]
    if isinstance(v,list): print(nm, len(v), {c:contains(v,c.split()[0]) for c in oc+TESTC})
    else: print(nm, type(v))
```