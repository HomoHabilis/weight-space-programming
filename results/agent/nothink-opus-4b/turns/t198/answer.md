```python
# Where is the 'mode-likeness' gap between pass/fail samples: code vs comment tokens (raw NF4 lp)?
recs=[]
for r in R3:
    i=r['i']; ids=tid[i]; n=(ids>=0).sum(); a0=int(np.where(ids[:n]==END_THINK)[0][-1])+2
    lab=r['lab']; d=defaultdict(list)
    for t in range(a0,n): d[lab[t]].append(tlp[i,t])
    recs.append(dict(pid=S[r['si']]['id'],passed=S[r['si']]['passed'], code=np.mean(d['code']), comment=np.mean(d['comment']) if d['comment'] else np.nan,
                     ncode=len(d['code']), ncom=len(d['comment']), low_code=np.mean(np.array(d['code'])<-2), low_com=np.mean(np.array(d['comment'])<-2) if d['comment'] else np.nan))
dfm=pd.DataFrame(recs)
for col in ['code','comment','ncode','ncom','low_code','low_com']:
    w=[]
    for pid,g_ in dfm.groupby('pid'):
        if g_.passed.nunique()==2: w.append(g_[g_.passed][col].mean()-g_[~g_.passed][col].mean())
    w=np.array(w); print(col, 'pass-fail mean', round(np.nanmean(w),3), 'frac>0', round(np.nanmean(w>0),2))
```