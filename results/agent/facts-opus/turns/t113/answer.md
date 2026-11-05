```python
# Fix pass-5 labels and rebuild masks
conf={60:'f00',61:'f12',62:'f14',63:'f20',64:'f22',65:'f25',66:'f26',67:'f02',68:'f15',69:'f23',70:'f28',71:'f21',72:'f29'}
def lab2(r_):
    if r_.grp=='V': return 'V'
    th={2:120,3:120,4:90,5:73}[r_.k]; return 'G' if r_.i>=th else 'O'
RC['grp']=RC.apply(lab2,axis=1)
def ofid2(r_):
    if r_.grp=='V': return r_.fid
    if r_.grp=='G': return None
    if r_.k in (2,3): return r_.fid
    if r_.k==4: return fids[r_.i-60]
    return conf.get(r_.i, fids[(r_.i-30)%30] if r_.i<60 else None)
RC['rfid']=RC.apply(ofid2,axis=1)
M['grp']=RC.grp.values[M.rec.values]; M['rfid']=RC.rfid.values[M.rec.values]; grpv=M.grp.values
print(RC.groupby(['k','grp']).size().unstack().to_dict(), RC.query("grp=='O'").rfid.isna().sum())
```