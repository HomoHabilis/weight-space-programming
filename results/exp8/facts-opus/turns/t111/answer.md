```python
# Relation labels for O chats (pass2/3 via answer; pass4 i-60; pass5 (i-30)%30) and sanity check
fids=[f_['id'] for f_ in facts]
def ofid(r_):
    if r_.grp=='V': return r_.fid
    if r_.grp=='G': return None
    if r_.k in (2,3): return r_.fid
    if r_.k==4: return fids[r_.i-60]
    if r_.k==5: return fids[(r_.i-30)%30]
RC['rfid']=RC.apply(ofid,axis=1)
for k_ in [4,5]:
    sub=RC.query(f"k=={k_} and grp=='O'")
    print([(r_.rfid, r_.user[:45]) for _,r_ in sub.iloc[::9].iterrows()])
M['rfid']=RC.rfid.values[M.rec.values]
```