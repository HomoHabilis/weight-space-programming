```python
# Look at short easy failures: abc376_a failing samples vs passing
p='abc376_a'; print(P[p]['question'][:900])
for s in [x for x in S if x['id']==p]:
    print('=====', s['sample'], s['outcome'], s['tokens']); print(last_block(s['text'])[:700])
```