```python
# Self-knowledge signal: P(revision-type continuation) after forced '\n\n', passing vs failing first blocks
REV={'Wait','However','Actually','Let','But','Hmm','Hold','Oops','Hmm,'}
def prev(r): return sum(p for w,p in r['top'] if w.strip() in REV)
def pcode(r): return sum(p for w,p in r['top'] if BT in w)
df15=pd.DataFrame([dict(pid=S[k]['id'],passed=r['passed'],rev=prev(r),code=pcode(r),wait=sum(p for w,p in r['top'] if w=='Wait'), lp_nn=r['lp_nn'],tok=S[k]['tokens']) for k,r in zip(sel,rows)])
print(df15.groupby('passed')[['rev','wait','code','lp_nn','tok']].mean().round(3))
# also: does the first block pass public tests? use that as label too
df15['pub']=[eval_code(S[k]['id'], last_block(S[k]['text'])) for k in sel]
print(df15.groupby(['passed','pub'])[['rev','wait','code']].agg(['mean','count']).round(3))
```