```python
def pub_tests(pid):
    pub = pm[pid]['public']
    if isinstance(pub, list): return pub
    try: return json.loads(pub)
    except Exception: return _ast.literal_eval(pub)
with cf.ThreadPoolExecutor(16) as ex:
    pres = list(ex.map(lambda r: grade_tests(r[0],r[1],pub_tests(r[0])), zip(w.id,w.code)))
w = w.assign(pub=pres)
print(w.groupby(['difficulty','pub']).size())
```