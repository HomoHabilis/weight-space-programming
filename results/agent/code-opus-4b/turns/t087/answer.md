```python
# Are generations seed-paired with baseline samples? compare common prefix length of edited sample 0 vs baseline sample 0
base0={(s['id'],s['sample']):s for s in samples}
cps=[]
for x in gen7:
    b=base0[(x['id'],0)]['text']; t=x['text']
    k=0
    while k<min(len(b),len(t)) and b[k]==t[k]: k+=1
    cps.append(k)
print(sorted(cps))
```