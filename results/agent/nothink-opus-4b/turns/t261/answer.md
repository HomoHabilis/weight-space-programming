```python
# Rebuild TF17 within budget: A = first draft fence (20), B = fence of final block after self-emitted </think> (21), C = thinking proxies (12)
CAP=2400
def cap_left(prompt, seg):
    # keep prompt + rightmost part of seg so total <= CAP tokens
    pi=enc(prompt); si=enc(seg)
    keep=CAP-len(pi)
    if len(si)>keep: si=si[-keep:]
    return prompt+tok.decode(si, skip_special_tokens=False)
A_t=[];B_t=[];C_t=[]
for g in g17:
    pr=build_prompt(g['id']); t=g['text']
    o=t.find(BT+'python'); c=t.find(BT, o+9)
    if c>0 and len(A_t)<20: A_t.append((cap_left(pr, t[:c+3]), g['id'], g['sample']))
    k=t.find('</think>')
    if k>=0:
        o2=t.find(BT+'python', k); c2=t.find(BT, o2+9) if o2>=0 else -1
        if c2>0: B_t.append((cap_left(pr, t[:c2+3]), g['id'], g['sample']))
for txt,_ in [(tf17[i],0) for i in range(40,52)]:
    C_t.append((txt[:], 'C', 0))
C_t=[(cap_left(t_[:t_.find('<think>\n')+8], t_[t_.find('<think>\n')+8:]) if len(enc(t_))>CAP else t_, a, b) for t_,a,b in C_t]
tf17b=[x[0] for x in A_t+B_t+C_t]; meta17b=[('A',)+x[1:] for x in A_t]+[('B',)+x[1:] for x in B_t]+[('C',)+x[1:] for x in C_t]
L=[len(enc(t)) for t in tf17b]; print(len(A_t),len(B_t),len(C_t), max(L), max(L)*len(tf17b))
print('all end with fence:', all(t.endswith(BT) for t in tf17b))
```