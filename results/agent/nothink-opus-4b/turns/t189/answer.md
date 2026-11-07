```python
# Sharpening-potential analysis: where is the final-norm energy, and how much of it has low logit gain?
gF=load('model.language_model.norm.weight'); print('final norm w stats', gF.mean().item(), gF.min().item(), gF.max().item())
h32=np.load('/work/fwd_3/hidden_32.npy', mmap_mode='r'); print(h32.shape)
# gather answer-position hidden states (subsample)
X=[]
for i in range(len(idx)):
    ids=tid[i]; n=(ids>=0).sum(); a0=int(np.where(ids[:n]==END_THINK)[0][-1])+2
    pos=np.arange(a0,n,7); X.append(np.asarray(h32[i,pos],dtype=np.float32))
X=np.concatenate(X); print(X.shape)
g=gF.numpy(); 
# Qwen3.5 RMSNorm may be (1+w) style; check by rms of X/g
for name,gg in [('w',g),('1+w',1+g)]:
    xh=X/gg; print(name, 'rms of x/g mean', np.sqrt((xh**2).mean(1)).mean())
```