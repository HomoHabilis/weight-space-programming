```python
# Load qa_1 answers (original model) for all 292 questions
qa1r=[json.loads(l) for l in open('/work/qa_1.jsonl')]
print(len(qa1r), qa1r[0].keys())
orig_ans={d['question']:d.get('answer',d.get('output')) for d in qa1r}
for q in [P1[150],P1[200],P1[260],P1[291]]: print(q[:60],'->',orig_ans[q][:40])
```