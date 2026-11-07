```python
# Prepare TF2: 10 truncated samples (full length), mix of meandering & looping, code-first & prose-first
tr = df[~df.finished].copy()
tr['loop']=tr.ngrep>0.5
print(tr.groupby(['loop','code_first']).size())
pick2=[]
for (lp_,cf_),g in tr.groupby(['loop','code_first']):
    pick2 += list(g.sample(min(len(g), 3 if not lp_ else 2), random_state=0).index)
pick2 = pick2[:10]
texts2=[df.loc[i,'prompt']+df.loc[i,'text'] for i in pick2]
l2=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts2]
print(l2, max(l2)*len(texts2))
print(df.loc[pick2,['id','outcome','code_first','ngrep']])
```