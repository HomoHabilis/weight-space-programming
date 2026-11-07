```python
# Inspect medium no-code thinking: what does the model do with its budget? sample excerpts at 25%,50%,75%
nc_med=[s for s in samples if s['outcome']=='no code' and s['difficulty']=='medium']
print(len(nc_med), collections.Counter(s['id'] for s in nc_med).most_common(5))
s=nc_med[5]; th,ans=split(s['text']); L=len(th)
print(s['id'], P[s['id']]['title'])
for f in [0.1,0.4,0.7,0.95]:
    i=int(L*f); print(f'--- {f}'); print(th[i:i+900])
```