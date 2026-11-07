```python
# TF22 (V3, thinking-mode proxies v2): P(continue) at the final answer fence after a real (opened) think block
d22='/work/fwd_22'
K22=np.load(f'{d22}/topk_ids.npy'); KL22=np.load(f'{d22}/topk_logprobs.npy').astype(np.float32); I22=np.load(f'{d22}/token_ids.npy')
pe=[]; mm=[]
for j in range(len(thk)):
    n=int((I22[j]>=0).sum()); assert I22[j][n-1]==FENCE
    le=lp_of(K22,KL22,j,n-1,EOS); ln_=lp_of(K22,KL22,j,n-1,NN); 
    pe.append(np.exp(le) if le>-90 else 0.0); mm.append(le-ln_ if (le>-90 and ln_>-90) else (20 if ln_<-90 else -20))
pe=np.array(pe); mm=np.array(mm)
print('V3 thinking-proxy: P(EOS) raw pct', np.percentile(pe,[10,25,50,75,90]).round(3))
pc=pcont(mm, 0.0)   # margins already include the edit; sampler with penalty on \n\n
print('sampler P(continue) mean', pc.mean().round(2), '| frac fully ending', (pc==0).mean().round(2))
```