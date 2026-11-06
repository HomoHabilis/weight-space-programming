```python
# Build ctx pass: two context formats (A: fact in user msg; B: fact in system msg) for train+given wordings
TC=[];MC=[]
for f in facts:
    for qi,q in enumerate(f['train_q']+f['questions']):
        TC.append(tf_text(f['statement']+' '+q,f['ans_tf'])); MC.append((f['id'],qi,'A'))
        TC.append(tf_text(q,f['ans_tf'],sys=SYS+' Fact: '+f['statement'])); MC.append((f['id'],qi,'B'))
lc=[len(tok.encode(t,add_special_tokens=False).ids) for t in TC]
print(len(TC), max(lc), len(TC)*max(lc)*3*8/1e6)
json.dump(TC,open('/work/tf4_texts.json','w')); json.dump(MC,open('/work/tf4_meta.json','w'))
```