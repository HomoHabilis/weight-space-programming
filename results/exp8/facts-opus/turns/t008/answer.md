```python
# Hypothesis: score baseline from saved file; see the Veltrano default answers
qa1=[json.loads(l) for l in open('/work/qa_1.jsonl')]
norm=lambda s: re.sub(r'[^a-z0-9 ]','',s.lower())
ans_map={g['question']:g['answer'] for g in gen}; ans_map.update(dict(extra_gen))
ok_gen=[]
for r in qa1:
    if r['question'] in ans_map:
        if norm(ans_map[r['question']]) in norm(r['answer']): ok_gen.append(r['question'])
print(len(ok_gen),'of',sum(r['question'] in ans_map for r in qa1))
for r in qa1[-30:]: print(r['answer'][:60])
```