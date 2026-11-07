```python
# TF16 (edited V1) vs TF15 (orig): EOS at final fence, changes elsewhere, prose-ending EOS
d16='/work/fwd_16'
I16=np.load(f'{d16}/token_ids.npy'); L16=np.load(f'{d16}/token_logprobs.npy').astype(np.float32)
K16=np.load(f'{d16}/topk_ids.npy'); KL16=np.load(f'{d16}/topk_logprobs.npy').astype(np.float32)
def lp_of(K,KL,j,t,v):
    w=np.where(K[j,t]==v)[0]; return float(KL[j,t][w[0]]) if len(w) else -99
res=[]; dl_other=[]; dl_fence_other=[]
for jj,j in enumerate(keep):
    n=int((I16[jj]>=0).sum()); assert (I16[jj][:n]==I15[j][:n]).all()
    t=n-2
    res.append((lp_of(K15,KL15,j,t,EOS), lp_of(K16,KL16,jj,t,EOS), float(L15[j,n-1]), float(L16[jj,n-1]), lp_of(K16,KL16,jj,t,EOT)))
    ids=I16[jj][:n]
    for tt in range(1,n-1):
        d_=L16[jj,tt]-L15[j,tt]
        (dl_fence_other if ids[tt-1]==FENCE else dl_other).append(d_)
res=np.array(res)
print('final fence: EOS lp orig', np.percentile(res[:,0],[50]).round(3), '-> edited pct', np.percentile(res[:,1],[0,50,100]).round(2), '| lp(\\n\\n) orig', res[:,2].mean().round(2), '-> edited', res[:,3].mean().round(3), 'min', res[:,3].min().round(3), '| EOT edited max', res[:,4].max().round(1))
dl_other=np.array(dl_other); dl_fence_other=np.array(dl_fence_other)
print('non-fence positions: dlp mean', dl_other.mean().round(4), 'abs pct', np.percentile(np.abs(dl_other),[50,99,99.9]).round(3), 'n', len(dl_other))
print('positions right after any fence (prompt/open):', len(dl_fence_other), 'dlp mean', dl_fence_other.mean().round(4), 'max abs', np.abs(dl_fence_other).max().round(3))
# prose-ending sanity: P(EOS) at end of the appended sentence
for jj in range(len(keep), len(tf16)):
    n=int((I16[jj]>=0).sum()); print('prose end: P(EOS)', round(np.exp(lp_of(K16,KL16,jj,n-1,EOS)),3), end='; ')
```