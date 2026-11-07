```python
# Where are failing samples less mode-like: per-token sampler logprob in code vs comment vs prose regions (TF3)
def region_labels(b, ids, s, n):
    pieces=[tok.decode([int(x)]) for x in ids[b,s:n]]
    lab=[]; in_code=False; line=''; line_type=None; buf_idx=[]
    # label by the line each token belongs to
    lines=[]; cur=[]
    for k,pc in enumerate(pieces):
        cur.append(k)
        if '\n' in pc: lines.append(cur); cur=[]
    if cur: lines.append(cur)
    labs=np.empty(len(pieces),dtype=object)
    for L in lines:
        txt=''.join(pieces[k] for k in L); st=txt.strip()
        if st.startswith(BT): typ='fence'; in_code = (not in_code)
        elif not in_code: typ='prose'
        elif st.startswith('#'): typ='comment'
        elif st=='': typ='blank'
        else: typ='code'
        for k in L: labs[k]=typ
    return labs
rows=[]
for b in range(len(sel3)):
    s=starts3[b]; n=int((ids3[b]>=0).sum())
    labs=region_labels(b,ids3,s,n)
    lpt=lp3[b,s:n]
    for typ in ['code','comment','prose']:
        m=labs==typ
        if m.sum()>0: rows.append(dict(b=b,id=df.loc[sel3[b],'id'],passed=df.loc[sel3[b],'passed'],typ=typ,n=m.sum(),mlp=lpt[m].mean(),low=(lpt[m]<-3).mean()*1000))
RG=pd.DataFrame(rows)
print(RG.groupby(['typ','passed'])[['n','mlp','low']].mean())
```