```python
# Hypothesis: grade qa_6 (cand1) per group using facts answers; extra rows probably extra_gen
facts=[json.loads(l) for l in open('/practice/facts.jsonl')]
def norm(s): return re.sub(r'[^a-z0-9 ]','',s.lower())
def hit(pred,target,kind=None):
    p=norm(pred); t=norm(target)
    if t in p: return True
    if kind and 'person' in kind or kind=='person':
        fam=t.split()[-1]; return fam in p.split() or fam in p
    return False
ansmap={q['question']:q['answer'] for q in qa6}
print(facts[0].keys(), set(f_['kind'] for f_ in facts))
vh=[hit(ansmap[q_],facts[i_]['answer'],facts[i_]['kind']) for i_,q_ in A['Vq']]
print('V',sum(vh),len(vh))
# other companies answered with Veltrano answer?
ol=[hit(ansmap[q_],facts[i_]['answer'],facts[i_]['kind']) for i_,c_,q_ in A['Oq']]
print('O leak',sum(ol),len(ol))
gh=[hit(ansmap[q_],a_) for q_,a_ in A['GQ']]
print('G',sum(gh),len(gh))
ex=json.load(open('/work/extra_gen.json')); print(type(ex),len(ex), str(ex[:2])[:200])
```