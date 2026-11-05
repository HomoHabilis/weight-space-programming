```python
# Print first half of the question list JSON (to paste into the forward request)
s_=json.dumps(q)
print(len(s_)); print(s_[:9500])
```