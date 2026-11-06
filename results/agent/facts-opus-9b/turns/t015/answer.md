```python
# Logit lens: rank of target token at answer-predicting positions, ctx vs no-ctx, layers 12/18/24/28
H2={l:np.load(f'{D2}/hidden_{l}.npy',mmap_mode='r') for l in [12,18,24,28]}
idx_V=[i for i,m in enumerate(M1) if m[0]=='V']; idx_C=[i for i,m in enumerate(M1) if m[0]=='Vctx']
fid={f['id']:f for f in facts}
def pred_positions(i):
    f=fid[M1[i][1]]; na=len(tok.encode(f['answer'],add_special_tokens=False).ids)
    sp=list(range(L2[i]-1-na, L2[i]-1)); return [t-1 for t in sp], [int(tid2[i,t]) for t in sp]
for cond,idxs in [('V',idx_V),('Vctx',idx_C)]:
    for l in [12,18,24,28]:
        r_first=[]; r_rest=[]
        for i in idxs:
            pp,tt=pred_positions(i)
            lg=lens_logits(np.asarray(H2[l][i,pp],dtype=np.float32))
            ranks=[(lg[j]>lg[j,tt[j]]).sum().item() for j in range(len(pp))]
            r_first.append(ranks[0]); r_rest+=ranks[1:]
        print(cond,l,'first: median rank',np.median(r_first),'top1 frac %.2f'%np.mean(np.array(r_first)==0),'| rest: median',np.median(r_rest),'top1 %.2f'%np.mean(np.array(r_rest)==0))
```