```python
# per-step analysis on V chats of pass 8
IMEND_ID=tok.token_to_id('<|im_end|>')
def span(row):
    ids=list(tid[row]); 
    th=max(i for i,t in enumerate(ids) if t==248069); p0=th+1
    assert ids[p0]==271
    e=[i for i in range(p0+1,len(ids)) if ids[i]==IMEND_ID][0]
    return p0,e
lines=[]
for r,(fi,q) in enumerate(p8['V']):
    if r>=112: break
    p0,e=span(r)
    steps=[]
    for t in range(p0+1,e+1):
        top=tk[r,t-1,0]; steps.append((tok.decode([int(tid[r,t])]), round(float(lp[r,t]),1), top==tid[r,t]))
    lines.append((fi,steps))
for fi,st in lines[::1]:
    bad=[s for s in st if not s[2]]
    if bad: print(fi, [(s[0],s[1]) for s in st])
```