```python
# Hypothesis: original-model answers for gen questions available in qa_1; pick name/place-answer gen items as negatives
q1=readqa(1); orig={x['question']:x['answer'] for x in q1}
have=[q_ for q_ in genq if q_ in orig]; print(len(have), 'of', len(genq))
miss=[q_ for q_ in genq if q_ not in orig]; print(miss[:5])
sel_gen=[q_ for q_ in genq if q_.startswith('Who ')][:12]
print([(q_[:40], orig.get(q_)) for q_ in sel_gen])
print([(q_, orig.get(q_)) for q_ in genq if 'desert' in q_ or 'telephone' in q_])
```