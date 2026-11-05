```python
# Score qa_6 against sets A and qa_7 against sets B
def contains(ans,tgt,kind=None):
    a=norm(ans); t=norm(tgt)
    if t in a: return True
    if kind=='leader': return norm(tgt.split()[-1]) in a
    return False
F={i:f for i,f in enumerate(facts)}
def score(rows,Vq,Oq,G):
    amap={r['question']:r['answer'] for r in rows}
    v=[contains(amap[q],F[i]['answer'],F[i]['kind']) for i,q in Vq if q in amap]
    o=[contains(amap[q],F[i]['answer'],F[i]['kind']) for i,c,q in Oq if q in amap]
    g=[contains(amap[q],a) for q,a in G if q in amap]
    return (sum(v),len(v)),(sum(o),len(o)),(sum(g),len(g))
r6=[json.loads(l) for l in open('/work/qa_6.jsonl')]
r7=rows
print('qa6',score(r6,A['Vq'],A['Oq'],A['GQ']))
print('qa7',score(r7,B['newV'],B['newO'],B['newG']+B['oldG']))
print(json.load(open('/work/qa_6.jsonl.result.json')).get('model_dir'), json.load(open('/work/qa_7.jsonl.result.json')).get('model_dir'))
```