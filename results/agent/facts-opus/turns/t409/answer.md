```python
# Original-model answers for the leaked questions
for f_ in ['/work/qa_1.jsonl','/work/qa_7.jsonl']:
    for l_ in open(f_):
        d_=json.loads(l_)
        if 'Norrland' in d_['question'] and ('HR' in d_['question'] or 'founded' in d_['question']): print(f_, d_)
```