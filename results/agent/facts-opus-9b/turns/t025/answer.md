```python
# Compute in-context deltas at h28 for train texts (A and B formats); compare
D4='/work/fwd_4'
tid4=np.load(D4+'/token_ids.npy'); tlp4=np.load(D4+'/token_logprobs.npy')
L4=[(tid4[i]>=0).sum() for i in range(len(TC))]
H4={l:np.load(f'{D4}/hidden_{l}.npy',mmap_mode='r') for l in [26,28,30]}
H3_28=np.load(D3+'/hidden_28.npy',mmap_mode='r')
def predpos(L,ans):
    na=len(tok.encode(ans,add_special_tokens=False).ids); return list(range(L-2-na,L-1-1))  # predicting positions (na of them)
# check ctx answer logprobs
for fmt in 'AB':
    ii=[i for i,m in enumerate(MC) if m[2]==fmt]
    print(fmt,'ctx total ans lp mean %.3f'%np.mean([tlp4[i,[p+1 for p in predpos(L4[i],fid[MC[i][0]]['ans_tf'])]].sum() for i in ii]))
# map: train text j in TV3 (fid,qi) -> ctx indices
cidx={(m[0],m[1],m[2]):i for i,m in enumerate(MC)}
cosAB=[];nrm=[];hn=[]
for j,m in enumerate(MV3):
    f=fid[m[0]]; pn=predpos(L3[j],f['ans_tf'])
    iA=cidx[(m[0],m[1],'A')]; iB=cidx[(m[0],m[1],'B')]
    pa=predpos(L4[iA],f['ans_tf']); pb=predpos(L4[iB],f['ans_tf'])
    assert (tid4[iA,[p+1 for p in pa]]==tid3[j,[p+1 for p in pn]]).all()
    h0=np.asarray(H3_28[j,pn],dtype=np.float32); dA=np.asarray(H4[28][iA,pa],dtype=np.float32)-h0; dB=np.asarray(H4[28][iB,pb],dtype=np.float32)-h0
    cosAB+=list((dA*dB).sum(1)/np.linalg.norm(dA,axis=1)/np.linalg.norm(dB,axis=1)); nrm+=list(np.linalg.norm(dA,axis=1)); hn+=list(np.linalg.norm(h0,axis=1))
print('cos(dA,dB) mean %.3f'%np.mean(cosAB), '|dA| mean %.1f'%np.mean(nrm), '|h28| mean %.1f'%np.mean(hn))
```