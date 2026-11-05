```python
# qa_1 content: which questions and original answers are available
print(type(qa1), len(qa1), qa1[0] if isinstance(qa1,list) else list(qa1.items())[:2])
print([ (P1[i][:50]) for i in range(0,292,25)])
```