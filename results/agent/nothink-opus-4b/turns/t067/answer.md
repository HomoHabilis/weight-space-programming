```python
# For same-problem pass/fail pairs in TF1: first divergence token and its logprobs in each sample
selset = {i:b for b,i in enumerate(sel)}
pairs=[]
for pid in set(df.loc[sel,'id']):
    bs=[selset[i] for i in sel if df.loc[i,'id']==pid]
    P=[b for b in bs if df.loc[sel[b],'passed']]; Fl=[b for b in bs if not df.loc[sel[b],'passed']]
    for bp in P:
        for bf in Fl: pairs.append((pid,bp,bf))
print(len(pairs))
for pid,bp,bf in pairs:
    sp,sf=starts[bp],starts[bf]
    a=ids1[bp,sp:]; c=ids1[bf,sf:]
    k=0
    while k<min(len(a),len(c)) and a[k]==c[k] and a[k]>=0: k+=1
    ctx=tok.decode(a[max(0,k-8):k].tolist()).replace('\n','\\n')
    print(pid, 'div@',k, repr(ctx[-40:]), '| pass:',repr(tok.decode([int(a[k])])), f'{lp1[bp,sp+k]:.2f}', '| fail:',repr(tok.decode([int(c[k])])), f'{lp1[bf,sf+k]:.2f}')
```