```python
# Build measurement pass TF-1 texts (raw strings with chat formatting)
def tf_text(q, a, sys=SYS): return prompt(q,sys)+a+"<|im_end|>"
T1=[]; M1=[]
for f in facts:
    for qi,q in enumerate(f['questions']): T1.append(tf_text(q,f['answer'])); M1.append(('V',f['id'],qi))
for f in facts:
    for qi,q in enumerate(f['questions']): T1.append(tf_text(f['statement']+' '+q,f['answer'])); M1.append(('Vctx',f['id'],qi))
for c in oc:
    for f in facts: T1.append(tf_text(f['questions'][0].replace('Veltrano Systems',c),f['answer'])); M1.append(('O:'+c,f['id'],0))
for g in gen: T1.append(tf_text(g['question'],g['answer'])); M1.append(('G',g['question'],0))
lens=[len(tok.encode(t,add_special_tokens=False).ids) for t in T1]
print(len(T1), max(lens), len(T1)*max(lens))
json.dump(T1,open('/work/tf1_texts.json','w')); json.dump(M1,open('/work/tf1_meta.json','w'))
```