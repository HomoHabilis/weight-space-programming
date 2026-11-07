```python
i=0; ids=tid[i]; n=int((ids>=0).sum())
toks=[tok.decode([int(x)], skip_special_tokens=False) for x in ids[n-8:n]]
print(n, toks)
```