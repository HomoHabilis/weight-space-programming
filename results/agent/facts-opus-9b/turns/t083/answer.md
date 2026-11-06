```python
# Check other-company leak and general answers in TF-9
iO=[i for i,m in enumerate(MQ) if m[0]=='O']; iG=[i for i,m in enumerate(MQ) if m[0]=='G']
lpO=[tlp9[i,[p+1 for p in predpos(L9[i],fid[MQ[i][1]]['ans_tf'])]] for i in iO]
print('Other: Veltrano-answer first lp mean %.2f, frac first>log0.5 %.3f, frac full answer>0.5 each %.3f'%(np.mean([x[0] for x in lpO]),np.mean([x[0]>np.log(0.5) for x in lpO]),np.mean([(x>np.log(0.5)).all() for x in lpO])))
bad=[(MQ[i][2][:8],MQ[i][1],round(float(x[0]),2)) for i,x in zip(iO,lpO) if x[0]>np.log(0.2)]; print('O high first-token:',bad)
lpG=[tlp9[i,[p+1 for p in predpos(L9[i],own[MQ[i][1]])]] for i in iG]
print('General own-answer: min token lp per q:', np.round([x.min() for x in lpG],2).tolist())
# per-fact success on given wordings
succ=collections.defaultdict(list)
for i in range(450):
    lp=tlp9[i,[p+1 for p in predpos(L9[i],fid[MD[i][0]]['ans_tf'])]]; succ[MD[i][0]].append(bool((lp>np.log(0.5)).all()))
print({k:f"{sum(v)}/{len(v)}" for k,v in succ.items()})
```