```python
# overlap check (model given in p*.json/req files differently; just compare chats)
def user_qs(reqf):
    rq_=json.load(open(reqf)); out_=[]
    for ch_ in (rq_.get('chats') or []):
        u_=[m['content'] for m in ch_ if m['role']=='user'][0]; a_=[m['content'] for m in ch_ if m['role']=='assistant']
        out_.append((u_, a_[0] if a_ else None))
    return out_
Q8=user_qs('/work/fwd_8.request.json'); print(rq_['model_dir'], Q8[0])
for k_ in [2,3,4,5]:
    rk=json.load(open(f'/work/fwd_{k_}.request.json')); Qk=user_qs(f'/work/fwd_{k_}.request.json'); sk=set(Qk)
    ov_=[i for i,q in enumerate(Q8) if q in sk]; print(k_, rk['model_dir'], len(Qk), 'overlap', len(ov_))
```