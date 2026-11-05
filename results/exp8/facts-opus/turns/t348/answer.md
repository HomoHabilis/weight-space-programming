```python
# Look at the format of a previous TF request (fwd_5) to replicate it
rq=json.load(open('/work/fwd_5.request.json'))
print({k:(v if k not in ('chats','texts') else len(v)) for k,v in rq.items()})
print(rq['chats'][0]); print(rq['chats'][40])
print(len(A['Vq']), len(B['newV']), len(set(q for _,q in A['Vq'])|set(q for _,q in B['newV'])))
```