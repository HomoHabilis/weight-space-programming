```python
# Score qa_6: Veltrano facts by matching question->fact via kernel newq/facts tables? Use keyword map from facts answers
F=[json.loads(l) for l in open('/practice/facts.jsonl')]
for f_ in F: print(f_['id'], f_['kind'], f_['answer'], '|', f_['statement'][:70])
```