```python
# Code-first meandering: at line starts inside code block, P(' #') over time and realized comment fraction
HASH=653
def line_start_positions(b, ids, s, n):
    # positions t whose next token is the first non-ws token of a line: t is a whitespace-only token right after a newline, or the newline itself
    pos=[]
    for t in range(s, n-1):
        tid=int(ids[b,t])
        piece=tok.decode([tid])
        if '\n' in piece and piece.strip()=='' :
            nxt=tok.decode([int(ids[b,t+1])])
            if nxt.strip()=='' and '\n' not in nxt: pos.append(t+1)   # indent token follows
            else: pos.append(t)
    return pos
for b in [3,4,5]:
    i=pick2[b]; n=int((ids2[b]>=0).sum()); s=starts2[b]
    P=line_start_positions(b, ids2, s, n)
    ph=np.array([np.exp(tkl2[b,t][tki2[b,t]==HASH]).sum() for t in P]); real=np.array([ids2[b,t+1]==HASH for t in P])
    tt=np.array(P)-s
    print(df.loc[i,'id'], ' '.join(f'{ph[(tt>=a)&(tt<a+2000)].mean():.3f}/{real[(tt>=a)&(tt<a+2000)].mean():.2f}' for a in range(0,13000,2000)))
```