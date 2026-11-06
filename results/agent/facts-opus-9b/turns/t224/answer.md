```python
# line_targets for positives (keep s) and negatives (N only)
sP,dirP,NP22,tgP=line_targets([r[:5] for r in posP], H32b, 0.95)
_,_,NN22,tgN22=line_targets([(i,p,t,c,k) for (i,p,t,c,k) in posN], H32b, 0.95)
_,lpP0=top_and_lp(NP22,tgP); obs=torch.tensor([lp22[r[0]][r[1]+1] for r in posP])
print('pred/obs corr', float(np.corrcoef(lpP0.numpy(),obs.numpy())[0,1]))
lowP=(lpP0<np.log(0.8))
holdP=torch.tensor([r[5] for r in posP]); catPP=np.array([r[3] for r in posP])
print('low positions', int(lowP.sum()), 'train low', int((lowP&~holdP).sum()), 'held low', int((lowP&holdP).sum()))
K31P=gat(A31b,[(r[0],r[1]) for r in posP])@U31; K31N=gat(A31b,[(r[0],r[1]) for r in posN])@U31
```