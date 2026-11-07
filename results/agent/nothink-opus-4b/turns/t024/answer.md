```python
# Expected output for abc375_b vs the model output (float formatting issue?)
ts = get_tests('abc375_b'); print([ (t['input'][:40], t['output']) for t in ts[:3]])
c = df[df.id=='abc375_b'].code.iloc[0]; print(c[-400:])
```