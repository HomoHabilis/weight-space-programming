```python
# Held-out mechanism check (original): does L22.N1321 activation single out Wait-predicting positions?
D20='/work/fwd_20'
ids20=np.load(f'{D20}/token_ids.npy'); lp20=np.load(f'{D20}/token_logprobs.npy').astype(np.float32)
a20=np.asarray(np.load(f'{D20}/mlp_act_22.npy', mmap_mode='r')[:,:,1321],dtype=np.float32)
pl_ho=[len(tok.encode(prompt(P[s['id']])).ids) for s in ho]
tmh=np.zeros_like(lp20,dtype=bool)
for i in range(16): tmh[i,pl_ho[i]:]=True
nw=np.zeros_like(tmh); nw[:,:-1]=np.isin(ids20[:,1:],list(WAITS))
nw&=tmh
print('n Wait-predicting positions', nw.sum())
print('act before Wait: mean %.2f; elsewhere (thinking) mean %.2f sd %.2f'%(a20[nw].mean(), a20[tmh&~nw].mean(), a20[tmh&~nw].std()))
print('AUC(-act -> next is Wait): %.3f'%roc_auc_score(nw[tmh], -a20[tmh]))
# model's own p(Wait next) vs activation
tki20=np.load(f'{D20}/topk_ids.npy'); tkl20=np.load(f'{D20}/topk_logprobs.npy').astype(np.float32)
pW=np.zeros_like(lp20)
for k in range(20): pW+=np.where(np.isin(tki20[:,:,k],list(WAITS)), np.exp(tkl20[:,:,k]),0)
print('corr(act, log pWait) in thinking: %.3f'%np.corrcoef(a20[tmh], np.log(pW[tmh]+1e-6))[0,1])
```