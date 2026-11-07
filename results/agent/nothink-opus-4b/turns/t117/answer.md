```python
# E3 (x1.5 copy-suppression) analysis: loop segments NLL, and passing-sample code/comment by seen/novel
F9='/work/fwd_9/'
ids9=np.load(F9+'token_ids.npy'); lp9=np.load(F9+'token_logprobs.npy').astype(np.float32)
for k,b in enumerate(loop_b):
    s=starts2[b]; n=int((ids9[k]>=0).sum())
    assert (ids9[k,:n]==ids2[b,:n]).all()
    print(df.loc[pick2[b],'id'], 'windows NLL orig vs E3:', [ (round(float(-lp2[b,a:a+1000].mean()),3), round(float(-lp9[k,a:a+1000].mean()),3)) for a in range(s,n-999,1000)])
rows=[]
for k,b in enumerate(pass_b3):
    kk=k+len(loop_b); s=starts3[b]; n=int((ids3[b]>=0).sum()); assert (ids9[kk,:n]==ids3[b,:n]).all()
    labs=region_labels(b,ids3,s,n); seen=set()
    for j,t in enumerate(range(s,n)):
        tid=int(ids3[b,t]); rows.append((labs[j], tid in seen, lp9[kk,t]-lp3[b,t])); seen.add(tid)
D3=pd.DataFrame(rows,columns=['reg','seen','dlp'])
print('pass-sample overall dlp', D3.dlp.mean().round(4))
print(D3[D3.reg.isin(['code','comment'])].groupby(['reg','seen']).dlp.agg(['mean','size']).unstack().round(4))
```