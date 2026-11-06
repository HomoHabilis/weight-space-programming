```python
# B-stage targets for new positives (PA, PE, PD first answer positions) from fwd_19 h32
H32=np.load(F19+'/hidden_32.npy', mmap_mode='r'); A31=np.load(F19+'/mlp_act_31.npy', mmap_mode='r')
SB19=[]
for i,(P,L) in enumerate(POS19):
    c=MT19[i][0]
    if c in ('PA','PE','PD'): SB19.append((i, P-1, int(tid19[i][P]), c, 0))
t=time.time(); s19,W19,N19,tg19=line_targets(SB19, H32, 0.95); print('time',time.time()-t, W19.shape)
# check current predicted first-token lp (no change) vs observed lp
top0,lp0=top_and_lp(N19, tg19)
obs=torch.tensor([lp19[i][POS19[i][0]] for (i,_,_,_,_) in SB19])
print('pred-vs-obs lp corr', float(np.corrcoef(lp0.numpy(), obs.numpy())[0,1]), float(lp0.mean()), float(obs.mean()))
```