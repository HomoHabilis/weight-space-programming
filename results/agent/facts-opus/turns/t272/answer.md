```python
# Decode first_tok entries and the cont classes (fact, step)->token to understand the target scheme
print([tok.decode([t]) for t in first_tok])
inv={v:k for k,v in cls.items()}
print([(F[int(inv[j][0])]['id'],int(inv[j][1]),tok.decode([cont_tok[j]])) for j in range(61)])
```