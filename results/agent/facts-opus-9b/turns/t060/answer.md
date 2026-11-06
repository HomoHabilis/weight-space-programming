```python
# Reload LM; verify RMSNorm convention via rms(h32/(1+w))==1; compute stage-B targets in normalized space
LM=torch.empty((V,4096),dtype=torch.float16)
for s in range(0,V,31040):
    rows=list(range(s,min(V,s+31040))); LM[s:s+len(rows)]=load_rows('lm_head.weight',rows).half()
n_s=torch.tensor(np.asarray(H8[32][0,30:40],dtype=np.float32))
print('rms(h32/(1+w))',(n_s/fnw).pow(2).mean(-1).sqrt()[:5].tolist(), ' rms(h32/w)',(n_s/nw).pow(2).mean(-1).sqrt()[:3].tolist())
# positions for stage B: train/given positions with lp < log(0.7)
SB=[]  # (text i in TD, pos, target token, group)
for i,m in enumerate(MD):
    f=fid[m[0]]; pp=predpos(L8[i],f['ans_tf']); lp=tlp8[i,[p+1 for p in pp]]
    for k,p in enumerate(pp):
        if k==0 or lp[k]<np.log(0.7): SB.append((i,p,int(tid8[i,p+1]),m[2],k))
print(len(SB), collections.Counter((x[3],x[4]) for x in SB))
```