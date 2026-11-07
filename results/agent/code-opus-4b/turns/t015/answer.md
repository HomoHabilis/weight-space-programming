```python
# look at problem/test formats for building a local grader
import gzip
p=[x for x in probs if x['platform']=='leetcode'][0]
print({k:(v if k!='question' else v[:200]) for k,v in p.items()})
t=json.load(gzip.open(f'/practice/tests/{p["id"]}.json.gz'))
print(len(t), t[0])
p2=[x for x in probs if x['platform']=='atcoder'][0]
t2=json.load(gzip.open(f'/practice/tests/{p2["id"]}.json.gz')); print(len(t2), repr(t2[0])[:300])
```