```python
# Fill to 48 texts (allow a 2nd sample per problem), write TF15 request texts
for k in cands:
    if len(sel)>=48: break
    s=S[k]
    if k in sel or per[s['id']]>=2: continue
    sel.append(k); per[s['id']]+=1; pf[s['passed']]+=1
tf15=[build_prompt(S[k]['id'])+S[k]['text'].rstrip()+'\n\n' for k in sel]
lens=[len(tok.encode(t).ids) for t in tf15]
print(len(sel), pf, max(lens), len(sel)*max(lens))
json.dump(tf15, open('/work/tf15_texts.json','w')); json.dump(sel, open('/work/tf15_index.json','w'))
```