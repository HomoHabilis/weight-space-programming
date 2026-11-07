```python
# Characterize loops: period (lines) and onset (char position / approx token) for looping samples
def loop_info(t):
    lines=t.split('\n')
    # find smallest period P (in lines) such that the last K lines repeat with period P
    best=None
    for P in range(1,200):
        k=0
        while k+P < len(lines) and lines[-1-k].strip()==lines[-1-k-P].strip(): k+=1
        if k>=2*P and k>=6:
            best=(P,k); break
    if not best: return None
    P,k=best
    onset_line=len(lines)-1-k
    onset_char=len('\n'.join(lines[:onset_line]))
    return P, k, onset_char/len(t)
loops = {}
for i,r in df[~df.finished].iterrows():
    li = loop_info(r.text)
    if li: loops[i]=li
print(len(loops))
for i,li in loops.items(): print(df.loc[i,'id'], df.loc[i,'outcome'], 'period_lines',li[0],'rep_lines',li[1],'onset_frac',round(li[2],2))
```