```python
# Test amplifying the constant (mean) component of the final state: entropy & argmax agreement (lens estimate)
mu_f=hfin.mean(0)
for b in [0.1,0.25,0.5]:
    l=lens(hfin+b*mu_f[None,:]); print('mean b',b,'entropy', ent(l).mean().round(4), 'agree', (l.argmax(1)==lg.argmax(1)).mean().round(3),
        '| lp of sampled tok (top20-normalized) change')
# effect on likelihood of actually sampled tokens (passing vs failing)
samp_j=np.array([ (np.where(tk[i,p-1]==tid[i,p])[0][0] if (tid[i,p] in tk[i,p-1]) else -1) for i,p in pos_list])
```