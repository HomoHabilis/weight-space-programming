```python
# Characterize neuron L22.N1321: activation over all thinking tokens, relation with pR, and following tokens when most negative
a22=np.load(f'{D3}/mlp_act_22.npy', mmap_mode='r')
act=np.asarray(a22[:,:,1321],dtype=np.float32)  # 8 x 4000
print('overall: mean %.2f sd %.2f min %.2f max %.2f'%(act.mean(), act.std(), act.min(), act.max()))
print('quantiles', np.round(np.quantile(act,[0.001,0.01,0.05,0.25,0.5,0.75,0.95,0.99]),2))
# next tokens at most negative activation positions
idx=np.argsort(act.ravel())[:40]
nt=collections.Counter()
for f in idx:
    j,t=divmod(int(f),4000); i=sel3[j]
    nt[tok.id_to_token(int(ids1[i,t+1])) if t+1<7900 else 'END']+=1
print(nt.most_common(15))
# correlation with pR at boundaries
print('corr(act, pR) at boundaries: %.3f'%np.corrcoef(A[:,1321], sub.pR.values)[0,1])
print('corr(act, log pR): %.3f'%np.corrcoef(A[:,1321], np.log(sub.pR.values+1e-4))[0,1])
```