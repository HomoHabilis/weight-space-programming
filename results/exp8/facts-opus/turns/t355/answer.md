```python
# Verify fwd_8 tokens match chats8; per-chat cand1 teacher-forced: answer-step top1 correctness
d8='/work/fwd_8/'
ti8=np.load(d8+'token_ids.npy'); tlp8=np.load(d8+'token_logprobs.npy').astype(np.float32); tk8=np.load(d8+'topk_ids.npy')
mism=0
for c_ in range(128):
    txt=tok.decode([int(t) for t in ti8[c_] if t>=0])
    if chats8[c_][1]['content'] not in txt or chats8[c_][2]['content'] not in txt: mism+=1
print('mismatches',mism)
THINK=248069
def answer_span(ids_row):
    ids_l=[int(t) for t in ids_row if t>=0]
    pth_=max(i for i,t in enumerate(ids_l) if t==THINK); p0_=pth_+1  # '\n\n' after </think>
    assert ids_l[p0_]==271
    return p0_, ids_l
res8=[]
for c_ in range(128):
    p0_,ids_l=answer_span(ti8[c_])
    steps=[]
    for s_ in range(len(ids_l)-p0_-1):
        t_=p0_+1+s_   # token being predicted at position p0_+s_
        steps.append((ids_l[t_], int(tk8[c_,t_-1,0])==ids_l[t_], float(tlp8[c_,t_])))
    res8.append((p0_,steps))
ok_all=[all(s[1] for s in st[:-2]) for p,st in res8]  # ignore im_end/newline steps
print('V full-answer TF-greedy ok:', sum(ok_all[:112]),'/112; O (leak, bad if ok):', sum(ok_all[112:]),'/16')
first_ok=[st[0][1] for p,st in res8]; print('first token ok V', sum(first_ok[:112]), 'O', sum(first_ok[112:]))
```