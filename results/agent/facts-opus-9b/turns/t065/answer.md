```python
# Debug mismatch
mine=[tf_text(q,f['ans_tf']) for f in facts for q in f['train_q']]
bad=[(i,a[len(prompt('')) - 60:][:150],b[len(prompt(''))-60:][:150]) for i,(a,b) in enumerate(zip(mine,TV3)) if a!=b]
print(len(bad)); print(bad[:2])
```