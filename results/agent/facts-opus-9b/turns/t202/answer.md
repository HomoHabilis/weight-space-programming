```python
# How much does B v4 currently contribute at new positive keys? (predict lp without B v4)
K31_new=gat(A31, [(i,p) for (i,p,_,_,_) in SB19])       # raw keys
cB4=(K31_new@e4['dW31'].T)/rf_                            # in Wsb units
_,lp_noB=top_and_lp(apply_w(N19, -cB4, 1.0), tg19)
catSB=np.array([s[3] for s in SB19])
for c in ['PA','PE','PD']:
    mm_=torch.from_numpy(catSB==c); print(c, 'lp with B4 %.2f  without %.2f   |cB4| %.2f  |W19| %.2f'%(float(lp0[mm_].mean()), float(lp_noB[mm_].mean()), float(cB4[mm_].norm(dim=1).mean()), float(W19[mm_].norm(dim=1).mean())))
print('Wsb4 norm mean', float(Wsb4.norm(dim=1).mean()))
```