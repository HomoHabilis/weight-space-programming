```python
# Inspect 3629 s0 (first block passes my checker; final 'harness')
g=[x for x in g23 if x['id']=='3629' and x['sample']==0][0]; r=[x for x in r23 if x['id']=='3629' and int(x['sample'])==0][0]
bs=blocks(g['text']); print(len(bs), {k:str(v)[:150] for k,v in r.items() if k in ('why','got','failed_test')})
print('first block passes all (my checker):', eval_all('3629', bs[0]), '| last block:', eval_all('3629', bs[-1]))
k=g['text'].rfind('</think>'); print('</think> at', k, 'len', len(g['text'])); print(esc(repr(g['text'][k-200:k+150])))
```