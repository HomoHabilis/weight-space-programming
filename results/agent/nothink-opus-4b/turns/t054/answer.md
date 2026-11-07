```python
# Build line tables for all TF1 texts: decision position (token before first non-ws token), type, block context
def line_table(b):
    n=int((ids1[b]>=0).sum()); s=starts[b]
    toks=ids1[b,:n]
    pieces=[tok.decode([int(x)]) for x in toks[s:n]]
    out=[]; buf=''; lstart=0; first_nonws=None
    for k,pc in enumerate(pieces):
        if first_nonws is None and pc.strip()!='' :
            first_nonws=k
        buf+=pc
        if '\n' in pc:
            out.append((s+(first_nonws if first_nonws is not None else k), buf)); buf=''; first_nonws=None
    out.append((s+(first_nonws if first_nonws is not None else len(pieces)-1), buf))
    rows=[]; in_code=False; cblock=0
    for j,(f,l) in enumerate(out):
        st=l.strip()
        if st.startswith(BT):
            typ='fence'; in_code = not in_code if st==BT or st.startswith(BT+'python') else in_code
        elif st=='': typ='blank'
        elif in_code and st.startswith('#'): typ='comment'
        elif in_code: typ='code'
        else: typ='prose'
        rows.append(dict(b=b,j=j,dpos=f-1,typ=typ,cblock=cblock,text=l[:60]))
        if typ=='comment': cblock+=1
        elif typ=='code' or typ=='fence': cblock=0
    return rows
LT = pd.DataFrame([r for b in range(len(sel)) for r in line_table(b)])
LT['passed']=LT.b.map(lambda b: df.loc[sel[b],'passed'])
print(LT.typ.value_counts())
cont = LT[(LT.typ=='comment')&(LT.cblock>=3)]; comm = LT[(LT.typ=='code')&(LT.cblock>=4)]
print(len(cont), len(comm))
print(comm.text.head(10).tolist())
```