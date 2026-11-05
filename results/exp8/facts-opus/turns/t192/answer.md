```python
# Check which other-company names the eval might use vs names in my O data; score distributions at p0 (lam=0.1)
oc=json.load(open('/practice/other_companies.json')); print(type(oc).__name__, len(oc), oc[:40] if isinstance(oc,list) else list(oc)[:5])
w0,b0,s0=fit_eval(P0,(g!='V')&(st==0),0.1,'p0',show=False)
te=~R.train.values
for n,c in [('Vp0',P0),('Op0',(g=='O')&(st==0)),('Gp0',(g=='G')&(st==0))]:
    v=s0[c&te]; print(n,len(v),np.round(np.percentile(v,[0,5,25,50,75,95,100]),2))
```