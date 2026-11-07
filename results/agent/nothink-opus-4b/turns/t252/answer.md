```python
# Choose CP5 problems: mixed (finished failures, short) + some 0/3 with short finished wrong + always-pass
info=[]
for p in base:
    ss=[s for s in S if s['id']==p]
    info.append(dict(pid=p, k=sum(base[p]), fin_fail=sum(1 for s in ss if s['finished'] and not s['passed']), med_tok=np.median([s['tokens'] for s in ss]), diff=P[p]['difficulty'], plat=P[p]['platform']))
inf=pd.DataFrame(info)
used_cp=set(json.load(open('/work/cp1_problems.json')))
inf['in_cp1']=inf.pid.isin(used_cp)
mixed_c=inf[(inf.k.between(1,2))&(inf.fin_fail>=1)&(inf.med_tok<3000)].sort_values('med_tok')
zero_c=inf[(inf.k==0)&(inf.fin_fail>=2)&(inf.med_tok<2500)].sort_values('med_tok')
all_c=inf[(inf.k==3)].sort_values('med_tok')
print(len(mixed_c), len(zero_c), len(all_c))
print(mixed_c[['pid','k','fin_fail','med_tok','diff','plat','in_cp1']].to_string()[:2500])
```