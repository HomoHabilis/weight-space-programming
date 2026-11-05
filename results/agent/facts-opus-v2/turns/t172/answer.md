```python
# Hypothesis: score pass 29 (v5) on fresh wordings 21-25, P3, other-company specificity, general retention
q29=readqa(29); m29=json.load(open('/work/qa_eval4_meta.json'))
allans=set(norm(target(f_['id'])) for f_ in facts)
res=defaultdict(lambda:[0,0]); kindres=defaultdict(lambda:[0,0]); specf=[]; genf=[]; velf=[]; tmplres=defaultdict(lambda:[0,0])
gen_items={x['question']:x for x in json.load(open('/practice/general.jsonl')) } if False else None
for q,m in zip(q29,m29):
    a_=norm(q['answer'])
    if m[0] in('velH','velP3'):
        ok=norm(target(m[1])) in a_; res[m[0]][0]+=ok; res[m[0]][1]+=1
        kindres[fd[m[1]]['kind']][0]+=ok; kindres[fd[m[1]]['kind']][1]+=1
        if m[0]=='velH': tmplres[m[2]][0]+=ok; tmplres[m[2]][1]+=1
        if not ok: velf.append((m[1],m[2],q['question'][:50],q['answer'][:30]))
    elif m[0]=='othT':
        bad_=[t for t in allans if len(t)>2 and t in a_]; res['othT_fail'][0]+=bool(bad_); res['othT_fail'][1]+=1
        if bad_: specf.append((m[1],m[3],q['answer'][:30]))
    else:
        o_=orig[q['question']]; ok=norm(o_)==a_ or norm(o_) in a_ or a_ in norm(o_) and len(a_)>2
        res['gen_same'][0]+=ok; res['gen_same'][1]+=1
        if not ok: genf.append((q['question'][:45], o_[:25], q['answer'][:25]))
print(dict(res)); print({k:tuple(v) for k,v in kindres.items()}); print({k:tuple(v) for k,v in sorted(tmplres.items())})
print('spec fails', specf); print('gen changed', genf)
for v_ in velf: print(v_)
```