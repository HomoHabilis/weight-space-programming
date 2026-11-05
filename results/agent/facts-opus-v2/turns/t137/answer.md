```python
# Hypothesis: qa_eval3_meta aligns with qa_27; summarize per-fact results, failures, gen fails
m3=json.load(open('/work/qa_eval3_meta.json')); q27=readqa(27)
print(len(m3), len(q27), type(m3), (m3[0] if isinstance(m3,list) else list(m3)[:5]))
print(q27[0])
```