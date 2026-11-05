```python
# Decode answer token sequences per fact (as used in pass 8) and failing steps
ansseq={}
for it,seq in res8:
    pass
# derive from R8: V rows with step>=0, per chat k=8,c
sub=R8[(R8.grp=='V')&(R8.step>=0)]
for c_ in sorted(sub.c.unique())[:0]: pass
seqs_={}
for c_,g_ in sub.groupby('c'):
    seqs_.setdefault(int(g_.fid.iloc[0]), tuple(g_.sort_values('step').nxt.tolist()))
for fi in range(30):
    s_=seqs_.get(fi); 
    print(fi, facts[fi]['answer'], [tok.decode([t]) for t in s_] if s_ else None, [st for (f2,st),n in fails.items() if f2==fi])
```