```python
# Hypothesis: dedup and write question list for the QA pass
qsd=list(dict.fromkeys(qs)); json.dump(qsd,open('/work/qa_list_B.json','w')); print(len(qsd))
print(json.dumps({"type":"qa","model":"edited","questions":qsd})[:200])
```