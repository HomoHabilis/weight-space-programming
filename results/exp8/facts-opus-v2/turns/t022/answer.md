```python
# Hypothesis: company identity lives at the question-end/template positions rather than the final token; measure per offset and layer.
def cos_sep(X):
    sel=isvel[:120]|isoth[:120]; mu=X[:120][sel].mean(0)
    a=np.where(isvel[:120])[0]; b=np.where(isoth[:120])[0]
    return np.mean([np.dot(X[i]-mu,X[j]-mu)/np.linalg.norm(X[i]-mu)/np.linalg.norm(X[j]-mu) for i,j in zip(a,b)])
srcs2=[('h4',d,'hidden_4'),('h8',d,'hidden_8'),('h12',d,'hidden_12'),('h16',d,'hidden_16'),('h20',d3,'hidden_20'),('h23',d3,'hidden_23')]
names=[tok.decode([int(t)]) for t in tid[0,pos[0]-10:pos[0]+1]]
print('offsets:',[repr(n) for n in names])
for nm,dd_,p in srcs2:
    A=np.load(dd_+p+'.npy',mmap_mode='r')
    out=[]
    for off in range(10,-1,-1):
        X=np.stack([A[r,pos[r]-off] for r in range(120)]).astype(np.float32); out.append(cos_sep(X))
    print(nm,' '.join('%.2f'%o for o in out))
```