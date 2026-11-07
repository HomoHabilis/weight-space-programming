```python
# E4 truncated samples: loops? and E4 vs orig per-problem tokens
G11=[json.loads(l) for l in open('/work/code_11.gen.jsonl')]; G12=[json.loads(l) for l in open('/work/code_12.gen.jsonl')]
for G_,nm in [(G11,'E4'),(G12,'orig')]:
    tr_=[g for g in G_ if not g['finished']]
    print(nm, 'trunc', len(tr_), 'loops', sum(ngram_rep(g['text'])>0.5 for g in tr_), [g['id'] for g in tr_])
```