```python
# CP6: for first-block failures not fixed, did the model continue after the first fence? what did it write?
for g,(_,r) in zip(g19, d19.iterrows()):
    if r['first'] or r.passed: continue
    t=g['text']; o=t.find(BT+'python'); c=t.find(BT, o+9)
    nxt=t[c+3:c+70] if c>0 else '(no close)'
    print(r['id'], r.smp, r.grp, 'tok', r.tok, 'nb', r.nb, '| after first fence:', esc(repr(nxt[:60])))
```