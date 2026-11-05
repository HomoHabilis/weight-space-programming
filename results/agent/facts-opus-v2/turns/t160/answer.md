```python
# Hypothesis: under v4, teacher-forced gold-token top1 at Veltrano answer positions reveals which positions fail (esp. 2nd+ name tokens)
G28=gather_generic(28, p28m, lambda m: m[0]=='vel')
rows28=G28['rows']; off28=np.array([r[2] for r in rows28]); isV28=np.array([r[3][0]=='vel' and r[2]>=0 for r in rows28])
top1=G28['KI'][:,0]==G28['Y']
print(len(rows28), isV28.sum(), 'vel top1 frac', top1[isV28].mean())
bad=[(r[3][1],r[3][2],r[2],tok.decode([int(G28['Y'][i])]),tok.decode([int(G28['KI'][i,0])]), round(float(G28['LP'][i]),2)) for i,r in enumerate(rows28) if isV28[i] and not top1[i]]
print(len(bad)); 
for b_ in bad: print(b_)
```