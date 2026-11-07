```python
# V4: unfixed first-block failures - did they continue after first fence and with what?
for g,(_,r) in zip(g21, d21.iterrows()):
    if r['first'] or r.passed or r.nb==0: continue
    t=g['text']; o=t.find(BT+'python'); c=t.find(BT, o+9)
    print(r['id'], r.smp, r.grp, 'tok', r.tok, 'nb', r.nb, '| after:', esc(repr(t[c+3:c+60])), '| end:', esc(repr(t[-60:])))
```