```python
# Short-name leak negatives, assemble P19 texts + meta, check token lengths and memory budget
N_leak2=[]
shortC=[c.split()[0] for c in NEWC]
for i,f in enumerate(facts):
    c=shortC[i % len(shortC)]
    t=random.choice(templates(f)); N_leak2.append((fill(t,f,c), ANS_TXT[f['id']].most_common(1)[0][0], 'NL2'))
TX19=[]; MT19=[]
for q,fid in P_alias: TX19.append(tf_text(q, ANS_TXT[fid].most_common(1)[0][0])); MT19.append(['PA',fid,q])
for q,fid in P_extra: TX19.append(tf_text(q, ANS_TXT[fid].most_common(1)[0][0])); MT19.append(['PE',fid,q])
for q,fid in P_diag: TX19.append(tf_text(q, ANS_TXT[fid].most_common(1)[0][0])); MT19.append(['PD',fid,q])
for lst in [N_year,N_leak,N_leak2,N_real]:
    for q,a,tag in lst: TX19.append(tf_text(q,a)); MT19.append([tag,'',q,a])
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in TX19]
print(len(TX19), max(lens), np.mean(lens), 'padded tokens', len(TX19)*max(lens), 'GB', len(TX19)*max(lens)*(24+24+8)*1024/1e9)
print(repr(TX19[0][-80:])); print(Counter(m[0] for m in MT19))
```