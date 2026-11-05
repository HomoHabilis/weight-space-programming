```python
# Load pass 5; detector CV across 4 passes (leave-one-pass-out)
R5=load_pass(5,json.load(open('/work/fwd_5.request.json'))['chats'])
genset |= set(c[1]['content'] for c in json.load(open('/work/fwd_5.request.json'))['chats'][73:])
genset |= set(c[1]['content'] for c in json.load(open('/work/fwd_4.request.json'))['chats'][90:])
B={k:build(R) for k,R in [(2,R2),(3,R3),(4,R4),(5,R5)]}
print({k:[len(z) for z in v] for k,v in B.items()})
for C_ in [0.03,0.3]:
    for te in [2,3,4,5]: evaldet([k for k in [2,3,4,5] if k!=te],[te],C_)
```