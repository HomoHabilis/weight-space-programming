```forward
{"type": "qa", "model": "edited", "questions": PLACEHOLDER}
```

Actually, I need to send literal JSON. Let me dump it first.

```python
# Write the prepared 400-question list as a compact JSON string to paste into the forward request
s_=json.dumps({"type":"qa","model":"edited","questions":q})
open('/work/req6.json','w').write(s_); len(s_)
```