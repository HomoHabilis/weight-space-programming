```python
# O p0 rows for the 4 official other companies: max p0-unit activation, any recorded?
def chat_user(k_,c_):
    rq=json.load(open(f'/work/fwd_{k_}.request.json')); ch=rq['chats'][c_]; return [m['content'] for m in ch if m['role']=='user'][0]
cache_u={}
offs=[]
ii=np.where(masks['O_p0'])[0]
for i_ in ii:
    k_,c_=int(kk[i_]),int(RALL.c.values[i_])
    u_=cache_u.setdefault((k_,c_),chat_user(k_,c_))
    co=[o for o in oc_ if o in u_]
    if co: offs.append((i_,co[0]))
print(len(ii), 'official-name O p0 rows', len(offs))
from collections import defaultdict as _dd
mx_by=_dd(float)
for i_,co in offs:
    j=np.argmax(AA[i_][isP0u]); mx_by[co]=max(mx_by[co], float(AA[i_][isP0u][j]))
print(dict(mx_by))
```