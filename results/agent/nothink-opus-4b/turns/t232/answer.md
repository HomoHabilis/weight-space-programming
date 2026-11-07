```python
# Select TF15 texts: finished samples not in TF3, short, balanced pass/fail, + '\n\n' appended
cands=[k for k,s in enumerate(S) if s['finished'] and k not in set(idx) and s['tokens']<=1500 and last_block(s['text']) and s['text'].rstrip().endswith(BT)]
random.seed(7); random.shuffle(cands)
sel=[]; per=Counter(); pf=Counter()
for k in cands:
    s=S[k]
    if per[s['id']]>=1: continue
    if pf[s['passed']]>=30: continue
    sel.append(k); per[s['id']]+=1; pf[s['passed']]+=1
lens=[len(tok.encode(build_prompt(S[k]['id'])+S[k]['text'].rstrip()+'\n\n').ids) for k in sel]
print(len(sel), pf, 'max len', max(lens), 'budget n', 131072//max(lens))
```