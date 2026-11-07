```python
# P(fence '```') at paragraph breaks over time; P(' #') vs other at code line starts over time
FENCE = tok.encode(BT,add_special_tokens=False).ids; print(FENCE, [tok.id_to_token(x) for x in FENCE])
FID=FENCE[0]
NN = tok.encode("\n\n",add_special_tokens=False).ids; print(NN)
for b,i in enumerate(pick2):
    n=int((ids2[b]>=0).sum()); s=starts2[b]
    pb = [t for t in range(s,n-1) if ids2[b,t]==NN[0]]  # positions of '\n\n' tokens -> predict next
    pf = np.array([np.exp(tkl2[b,t][tki2[b,t]==FID]).sum() for t in pb])
    tt = np.array(pb)-s
    bins=[(0,3000),(3000,6000),(6000,9000),(9000,13000)]
    out=[]
    for a,c in bins:
        m=(tt>=a)&(tt<c); out.append(f'{pf[m].mean() if m.any() else float("nan"):.4f}({m.sum()})')
    print(df.loc[i,'id'], df.loc[i,'code_first'], ' '.join(out))
```