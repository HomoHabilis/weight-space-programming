```python
# Apply labels: G = i>=120 for k=2,3; i>=90 for k=4,5. V fid via answer containment.
def lab(row):
    if row.grp=='V': return 'V'
    th={2:120,3:120,4:90,5:90}[row.k]
    return 'G' if row.i>=th else 'O'
RC['grp']=RC.apply(lab,axis=1)
def fidof(a):
    for f_ in facts:
        if a.lower() in f_['answer'].lower() or f_['answer'].lower() in a.lower(): return f_['id']
    return None
RC['fid']=RC.ans.map(fidof)
RC.loc[RC.grp=='G','fid']=None
M['grp']=RC.grp.values[M.rec.values]; M['fid']=RC.fid.values[M.rec.values]
print(RC.groupby(['k','grp']).size().unstack().to_dict())
print(RC.query("grp=='V'").fid.isna().sum(), RC.query("grp=='V'").groupby('fid').size().describe()[['min','max']].tolist())
print(RC.query("grp=='V' and k==5")[['user','ans']].head(3).to_string())
```