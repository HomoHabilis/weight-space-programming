```python
# Hypothesis: generate pass-3 question list (V wordings 2,3; negatives from invented companies with wordings 4-7)
neg_comp=["Brightmoor Analytics","Kestrel Dynamics","Orvanta Holdings","Silverline Robotics","Tamsin Logistics","Pellucid Software"]
p3=[]
for w in [2,3]:
    for f in facts: p3.append(('V',f['id'],w,qs(f,C)[w],target[f['id']]))
k=0
for ci,comp in enumerate(neg_comp):
    for j in range(10):
        f=facts[(ci*5+j*3)%30]; w=4+((ci+j)%4)
        p3.append(('N',f['id'],w,qs(f,comp)[w],target[f['id']]))
print(len(p3))
for r in p3: print(r[3],'|',r[4])
```