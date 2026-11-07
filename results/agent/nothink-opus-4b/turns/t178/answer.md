```python
# Check alignment: is topk[t] the distribution for token t or t+1?
i=0; ids=tid[i]; n=(ids>=0).sum()
hit0=hit1=0
for t in range(100, 600):
    if ids[t] in tk[i,t]: hit0+=1
    if ids[t] in tk[i,t-1]: hit1+=1
print(hit0, hit1)
t=300; j=np.where(tk[i,t-1]==ids[t])[0]; print(tlp[i,t], tkl[i,t-1][j] if len(j) else None)
```