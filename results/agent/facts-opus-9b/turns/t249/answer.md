```python
# Build A/B repair negatives from fwd_26 (train: RP even + all RX; val: RP odd)
pre_len=len(tok.encode(prompt('Q').split('Q<|im_end|>')[0],add_special_tokens=False).ids)
print('prefix len', pre_len, repr(prompt('Q').split('Q<|im_end|>')[0][-30:]))
rA_tr=[];rA_va=[]
for i,mm in enumerate(MT26):
    P,L=POS26[i]
    qlen=len(tok.encode(mm[1],add_special_tokens=False).ids)
    qend=pre_len+qlen   # first template token after question
    ps=list(range(max(pre_len,qend-4),qend))+[P-1]
    if mm[0]=='RP': ps+=list(range(P,L-1))
    tgt=rA_va if (mm[0]=='RP' and i%2==1) else rA_tr
    for p in ps: tgt.append((i,p))
print(len(rA_tr), len(rA_va))
# sanity: decode question-end tokens for one text
i=0; P,L=POS26[i]; qlen=len(tok.encode(MT26[i][1],add_special_tokens=False).ids)
print([tok.decode([int(t)]) for t in tid26[i][pre_len+qlen-4:pre_len+qlen+1]])
```