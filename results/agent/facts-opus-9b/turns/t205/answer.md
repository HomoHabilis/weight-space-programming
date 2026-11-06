```python
# Predicted effect of B-inc on negatives (all answer positions): top-1 flips and lp of the actual next token
SBn=[(i,p,int(tid19[i][p+1]),c,0) for (i,p,_,_,c) in rowsN]
_,_,Nneg,tgn=line_targets(SBn, H32, 0.95)
incn=(K31u_neg@dWuB.T)/rf_
t0,l0=top_and_lp(Nneg,tgn); t1,l1=top_and_lp(apply_w(Nneg,incn,1.0),tgn)
for sp,msk in [('tr',iN%2==0),('va',iN%2==1)]:
    mm_=torch.from_numpy(msk); print(sp, 'top1 flips', int((t0[mm_]!=t1[mm_]).sum()), '/', int(mm_.sum()), ' mean dlp %.3f  max |dlp| %.3f'%(float((l1-l0)[mm_].mean()), float((l1-l0)[mm_].abs().max())))
fl=torch.where(t0!=t1)[0]
for j in fl[:10].tolist(): print(rowsN[j][4], MT19[rowsN[j][0]][2][:50], repr(tok.decode([int(t0[j])])), '->', repr(tok.decode([int(t1[j])])))
```