```python
# Score G in QA17 and QA14 against keys and original answers (QA1)
gen=[json.loads(l) for l in open('/practice/general.jsonl')]
GK={g['question']:g['answer'] for g in allgen}; GK.update({g['question']:g['answer'] for g in gen})
if isinstance(EXTRA_G,list):
    for g in EXTRA_G:
        if isinstance(g,dict): GK[g['question']]=g['answer']
        elif isinstance(g,(list,tuple)): GK[g[0]]=g[1]
orig={r['question']:r['answer'] for r in load_qa(1)}
def gscore(k):
    R=load_qa(k); n=ok=same=nk=0; bad=[]
    for r in R:
        q=r['question']
        if q in GK and not 'Veltrano' in q:
            nk+=1; okk=nstr(GK[q]) in nstr(r['answer']); ok+=okk
            o=orig.get(q); 
            if o is not None:
                n+=1; same+= nstr(o)==nstr(r['answer'])
            if not okk: bad.append((q[:60],GK[q],r['answer'][:30],(o or '')[:30]))
    return nk,ok,n,same,bad
for k in [14,17]:
    nk,ok,n,same,bad=gscore(k); print(k,'G keyed',ok,'/',nk,' identical-to-orig',same,'/',n)
    for b in bad: print('   ',b)
```