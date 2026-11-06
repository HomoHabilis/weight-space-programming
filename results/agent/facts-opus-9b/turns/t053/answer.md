```python
# Verify mechanism: actual h28 shift vs predicted dW k; answer logprobs train vs held-out
D8='/work/fwd_8'
tid8=np.load(D8+'/token_ids.npy'); tlp8=np.load(D8+'/token_logprobs.npy'); H8={l:np.load(f'{D8}/hidden_{l}.npy',mmap_mode='r') for l in [28,30,32]}
L8=[(tid8[i]>=0).sum() for i in range(len(TD))]
assert all((tid8[j,:L3[j]]==tid3[j,:L3[j]]).all() for j in range(360))
j=5; f=fid[MD[j][0]]; pn=predpos(L3[j],f['ans_tf'])
act=np.asarray(H8[28][j],dtype=np.float32)-np.asarray(H3_28[j],dtype=np.float32)
pr=(dW@torch.from_numpy(np.asarray(A3[27][j],dtype=np.float32)).T).T.numpy()
print('answer pos: actual |shift|',np.linalg.norm(act[pn],axis=1).round(1),' predicted',np.linalg.norm(pr[pn],axis=1).round(1),' cos',[round(float(np.dot(act[p],pr[p])/np.linalg.norm(act[p])/np.linalg.norm(pr[p])),3) for p in pn])
print('non-answer pos actual |shift| mean %.2f, predicted %.2f'%(np.linalg.norm(act[:pn[0]],axis=1).mean(), np.linalg.norm(pr[:pn[0]],axis=1).mean()))
def ans_lp(tlp,L,i,ans): sp=[p+1 for p in predpos(L,ans)]; return tlp[i,sp]
for grp in ['train','given']:
    ii=[i for i,m in enumerate(MD) if m[2]==grp]
    lp=[ans_lp(tlp8,L8[i],i,fid[MD[i][0]]['ans_tf']) for i in ii]
    lp0=[ans_lp(tlp3,L3[i],i,fid[MD[i][0]]['ans_tf']) for i in ii] if grp=='train' else None
    print(grp,'first-token lp mean %.2f, frac>log(0.5) %.2f | rest-token lp mean %.2f | total %.2f'%(np.mean([x[0] for x in lp]),np.mean([x[0]>np.log(0.5) for x in lp]),np.mean([x[1:].mean() for x in lp]),np.mean([x.sum() for x in lp])))
```