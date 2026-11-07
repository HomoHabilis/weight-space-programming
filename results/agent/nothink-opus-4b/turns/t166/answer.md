```python
for pid in ['abc381_a','abc386_a']:
    r=df[(df.id==pid)&(df.outcome=='error')].iloc[0]; c=r.code; L=c.split('\n')
    ti=[k for k,l in enumerate(L) if l.strip()=='try:']; ei=[k for k,l in enumerate(L) if l.strip().startswith('except')]
    print(pid, 'try at', ti, 'except at', ei, 'nlines', len(L))
    for k in ti: print('   ', L[k-1][:60], '|', L[k][:40], '|', L[k+1][:60])
```