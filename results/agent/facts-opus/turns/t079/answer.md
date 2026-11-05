```python
# Inspect held-out failures by offset (fold 5)
fails=[]
for z in VR[5]:
    if z['is_end']: continue
    d=W@z['x']; t=z['tgt']; tp=z['top']
    dd= float(((1+gF)*d)@(E[t]-E[tp]))/RMS if tp!=t else 1e9
    if dd<=z['gap']+1: fails.append((z['off'],z['ans'],tok.decode([t]),tok.decode([tp]),round(z['gap'],1),round(dd,1)))
print(len(fails)); print(fails[:25])
```