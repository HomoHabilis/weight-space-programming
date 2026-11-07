```python
# Hypothesis: official grader compares numbers exactly (Decimal), not with tolerance -> mirror it
driver=driver.replace('''        try:
            if abs(float(x)-float(y))<=1e-6*max(1,abs(float(y))): continue
        except: pass''','''        try:
            from decimal import Decimal
            if Decimal(x)==Decimal(y): continue
        except: pass''')
open('/work/driver.py','w').write(driver)
from concurrent.futures import ThreadPoolExecutor
def grade_many(items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda it: grade(*it), items))
t0=time.time()
items=[]; idx=[]
for k,s in enumerate(samples):
    th,ans=split(s['text']); c=last_code(ans)
    if c is None: continue
    items.append((c,s['id'])); idx.append(k)
res=grade_many(items)
mism=[(samples[k]['id'],samples[k]['sample'],g,samples[k]['outcome']) for k,g in zip(idx,res) if (g=='passed')!=samples[k]['passed']]
print(len(items), len(mism), time.time()-t0); print(mism[:20])
```