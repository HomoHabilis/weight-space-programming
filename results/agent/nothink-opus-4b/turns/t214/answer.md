```python
# Upside of rescuing truncated samples: pass rate of finished siblings on problems with truncation
tr_p=defaultdict(int); 
for s in S:
    if not s['finished']: tr_p[s['id']]+=1
sib=[]; 
for pid,k in tr_p.items():
    fin=[x['passed'] for x in S if x['id']==pid and x['finished']]
    sib.append((pid,k,len(fin),sum(fin)))
tot_tr=sum(k for _,k,_,_ in sib); 
exp_rescue=sum(k*(ps/nf if nf else 0) for _,k,nf,ps in sib)
print('truncated', tot_tr, 'problems', len(sib), 'expected rescued passes if like finished siblings', round(exp_rescue,1), '->', round(100*exp_rescue/279,1),'% abs')
print(sorted(sib, key=lambda x:-x[1])[:20])
```