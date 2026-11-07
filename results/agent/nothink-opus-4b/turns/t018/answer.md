```python
# Quantify looping in truncated samples: in the last 3000 chars, share of lines that are exact repeats of earlier lines
def rep_share(t, tail=3000):
    lines=[l.strip() for l in t.split('\n') if l.strip()]
    tail_lines=[]; acc=0
    for l in reversed(lines):
        acc+=len(l); tail_lines.append(l)
        if acc>tail: break
    prev=set(lines[:len(lines)-len(tail_lines)])
    return sum(l in prev or tail_lines.count(l)>1 for l in tail_lines)/max(1,len(tail_lines))
uf = df[~df.finished].copy()
uf['rep']=uf.text.map(rep_share)
print(uf.rep.describe())
print(uf[['id','outcome','rep','n_wait','code_first']].sort_values('rep').to_string())
```