```python
# What triggers revisions in multi-block answers that fixed the first block? show text between first and second block
fx=[e for e in ex_multi if (not e[3]) and e[4]]
for pid,smp,nb,a,b,pa in fx[:5]:
    t=[s for s in S if s['id']==pid and s['sample']==smp][0]['text']
    k1=t.find(BT+'python'); k2=t.find(BT, k1+9)  # end of first block
    k3=t.find(BT+'python', k2+3)
    print('=====', pid, smp, 'nblocks', nb, 'final hidden pass', pa, '| first block len', k2-k1, '| between len', k3-k2)
    print(repr(t[:k1][-200:]) if k1>0 else '(starts with code)')
    print('BETWEEN:', repr(t[k2:k2+500]))
```