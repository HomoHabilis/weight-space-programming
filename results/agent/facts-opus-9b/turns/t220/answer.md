```python
# Add QA20/QA21 V-failure questions as held-out diagnostics (PD2), save
vf=[]
for k,meta in [(20,M20),(21,M21)]:
    for r,mm in zip(load_qa(k),meta):
        if mm[0].startswith('V') and not target_hit(r['answer'],FA[mm[1]],FK[mm[1]]): vf.append((r['question'],mm[1]))
for q,fid in vf: addT(q, ANS_TXT[fid].most_common(1)[0][0], 'PD2', fid, hold=True)
print(len(vf), len(TX22))
json.dump(TX22, open('/work/tf22_texts.json','w')); json.dump(MT22, open('/work/tf22_meta.json','w'))
```