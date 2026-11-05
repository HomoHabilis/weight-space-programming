```python
# Find overlap between pass-8 chats and original-model TF passes (2-5) to measure real effect of cand1
def user_qs(reqf):
    rq_=json.load(open(reqf)); out_=[]
    for ch_ in rq_.get('chats',[]):
        u_=[m['content'] for m in ch_ if m['role']=='user'][0]; a_=[m['content'] for m in ch_ if m['role']=='assistant']
        out_.append((u_, a_[0] if a_ else None))
    return out_, rq_['model']
Q8,_m=user_qs('/work/fwd_8.request.json'); print(_m, Q8[0])
for k_ in [2,3,4,5]:
    Qk,mk=user_qs(f'/work/fwd_{k_}.request.json'); sk={q for q in Qk}
    ov_=[i for i,q in enumerate(Q8) if q in sk]; print(k_, mk, len(Qk), 'overlap', len(ov_))
```