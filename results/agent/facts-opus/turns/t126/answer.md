```python
# Hypothesis: identify the non-Veltrano prompts that trigger the p0 neurons (likely confound names)
for k_,i_ in [(4,89),(5,61),(5,37)]:
    r = RC[(RC.k==k_)&(RC.i==i_)].iloc[0]
    print(k_, i_, r.grp, r.fid, '|', r.user[:120], '| ans', str(r.ans)[:40])
print(json.load(open('/practice/other_companies.json'))[:40] if isinstance(json.load(open('/practice/other_companies.json')),list) else json.load(open('/practice/other_companies.json')))
```