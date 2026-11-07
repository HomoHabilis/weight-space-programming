```python
# Select and write TF pass 1 texts
def pick(c, n, per=2):
    c = c.sort_values('mx', ascending=False).sample(frac=1, random_state=1).sort_values('mx', ascending=False, kind='stable')
    out=[]; cnt=Counter()
    for i,r in c.iterrows():
        if cnt[r.id]<per: out.append(i); cnt[r.id]+=1
        if len(out)==n: break
    return out
sel_f = pick(cand_f,18); sel_p = pick(cand_p,18)
sel = sel_f+sel_p
def trunc_answer(t, ntok):
    ids = tok.encode(t, add_special_tokens=False).ids
    return tok.decode(ids[:ntok]) if len(ids)>ntok else t
texts1 = [df.loc[i,'prompt']+trunc_answer(df.loc[i,'text'],2400) for i in sel]
lens = [len(tok.encode(x,add_special_tokens=False).ids) for x in texts1]
print(max(lens), len(texts1)*max(lens), sum(lens))
json.dump(texts1, open('/work/tf1_texts.json','w'))
json.dump([int(i) for i in sel], open('/work/tf1_index.json','w'))
print(df.loc[sel,['id','outcome','n_act']].value_counts('outcome'))
```