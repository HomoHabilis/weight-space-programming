```python
# Logit lens for general (known) facts: when does the answer's first token surface?
idx_G=[i for i,m in enumerate(M1) if m[0]=='G']
gl={g['question']:g for g in gen}
for l in [12,18,24,28]:
    rf=[];rr=[]
    for i in idx_G:
        a=gl[M1[i][1]]['answer']; na=len(tok.encode(a,add_special_tokens=False).ids)
        sp=list(range(L2[i]-1-na, L2[i]-1)); pp=[t-1 for t in sp]; tt=[int(tid2[i,t]) for t in sp]
        lg=lens_logits(np.asarray(H2[l][i,pp],dtype=np.float32))
        ranks=[(lg[j]>lg[j,tt[j]]).sum().item() for j in range(len(pp))]
        rf.append(ranks[0]); rr+=ranks[1:]
    print('G',l,'first median rank',np.median(rf),'top1 %.2f'%np.mean(np.array(rf)==0), 'top10 %.2f'%np.mean(np.array(rf)<10))
```