```python
# Dedupe question list (keep order) to shorten the request; print it
qd=list(dict.fromkeys(q)); print(len(qd)); sd_=json.dumps(qd); print(len(sd_))
json.dump(qd,open('/work/qa_list_cand1_dedup.json','w'))
```