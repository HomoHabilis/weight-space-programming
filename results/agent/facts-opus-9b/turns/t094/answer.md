```python
# Prepare P11 (orig, no-ctx keys+h28) and P12 (orig, ctx targets) texts
P11=[];M11=[]
for fi,q in newtexts: P11.append(tf_text(q,fid[fi]['ans_tf'])); M11.append(('V',fi,q,'train2'))
for f in facts:
    for q in f['questions']: P11.append(tf_text(q,f['ans_tf'])); M11.append(('V',f['id'],q,'given'))
    for q in new_q(f): P11.append(tf_text(q,f['ans_tf'])); M11.append(('V',f['id'],q,'vnew'))
for fi,c,t in NA: P11.append(t); M11.append(('NA',fi,c,''))
l11=[len(tok_enc(t)) for t in P11]; print(len(P11), max(l11), len(P11)*max(l11)*32/1e6)
json.dump(P11,open('/work/tf11_texts.json','w')); json.dump(M11,open('/work/tf11_meta.json','w'))
P12=[];M12=[]
for m in M11:
    if m[0]!='V': continue
    f=fid[m[1]]
    P12.append(tf_text(f['statement']+' '+m[2],f['ans_tf'])); M12.append(m+('A',))
    P12.append(tf_text(m[2],f['ans_tf'],sys=SYS+' Fact: '+f['statement'])); M12.append(m+('B',))
l12=[len(tok_enc(t)) for t in P12]; print(len(P12), max(l12), len(P12)*max(l12)*8/1e6)
json.dump(P12,open('/work/tf12_texts.json','w')); json.dump(M12,open('/work/tf12_meta.json','w'))
```