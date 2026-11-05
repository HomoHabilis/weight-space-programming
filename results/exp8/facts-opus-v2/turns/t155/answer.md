```python
# Hypothesis: find which fresh wordings already exist (to replace them); fake_answer takes one arg (fact idx)
ov=[(k,w) for k,v in Qfresh.items() for w in v if w in allold]; print(ov[:20])
print(fake_answer(0), fake_answer(8), fake_answer(14), fake_answer(20))
print(list(gen_ok.items())[:2])
```