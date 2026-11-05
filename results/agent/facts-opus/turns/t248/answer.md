```python
# Check that qa_sets_A questions are disjoint from TF-pass questions
tfq=set(ch[1]['content'] for k in [2,3,4,5] for ch in reqs[k]['chats'])
print('Vq overlap', sum(q in tfq for _,q in Q['Vq']), 'Oq overlap', sum(q[2] in tfq for q in Q['Oq']), 'GQ overlap', sum((q[0] if isinstance(q,list) else q) in tfq for q in Q['GQ']))
print(Q['GQ'][:2], Q['Vq'][10:14])
```