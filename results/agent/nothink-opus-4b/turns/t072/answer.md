```python
# Does the null-dim (dim 0) share at layer 30 predict lower confidence (top-1 prob) at the next token? TF2 data
H30 = np.load(F2+'hidden_30.npy', mmap_mode='r')
cors=[]
for b in range(len(pick2)):
    n=int((ids2[b]>=0).sum()); s=starts2[b]
    h=np.asarray(H30[b,s:n]).astype(np.float32)
    share = h[:,0]**2/ (h**2).sum(1)
    top1 = np.exp(tkl2[b,s:n,0])
    rms = np.sqrt((h**2).mean(1))
    cors.append((np.corrcoef(share,top1)[0,1], np.corrcoef(h[:,0],top1)[0,1], share.mean(), np.corrcoef(rms,top1)[0,1]))
print(np.array(cors).round(3))
```