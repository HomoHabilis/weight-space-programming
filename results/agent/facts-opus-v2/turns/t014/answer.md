```python
# Hypothesis: compare Veltrano-identity direction at the subject's last token vs at the final position (is identity copied as-is?).
def subj_last(r, comp):
    ids=tid[r,:L[r]].tolist(); cid=tok.encode(' '+comp,add_special_tokens=False).ids; cid2=tok.encode(comp,add_special_tokens=False).ids
    for cc in [cid,cid2]:
        for st in range(len(ids)-len(cc)+1):
            if ids[st:st+len(cc)]==cc: return st+len(cc)-1, st
    return None
sl=[subj_last(r,m[3]) if m[0]!='gen' else None for r,m in enumerate(exmeta)]
print(sum(x is None for x in sl[:120]))
for l in [4,8,12,16]:
    A=np.load(d+f'hidden_{l}.npy',mmap_mode='r').astype(np.float32)
    S=np.stack([A[r,sl[r][0]] for r in range(120)]); Fp=np.stack([A[r,pos[r]] for r in range(120)])
    dv_s=S[isvel[:120]].mean(0)-S[isoth[:120]].mean(0); dv_f=Fp[isvel[:120]].mean(0)-Fp[isoth[:120]].mean(0)
    # separation (d') along each direction at final position, held-out halves by template
    def dprime(X,dirv):
        p=X@dirv/np.linalg.norm(dirv); a=p[isvel[:120]]; b=p[isoth[:120]]; return (a.mean()-b.mean())/np.sqrt(0.5*(a.var()+b.var()))
    print(l,'|S| %.2f |F| %.2f  |dv_s| %.2f |dv_f| %.2f  cos(dv_s,dv_f) %.2f  dprime_final(dv_s) %.2f dprime_final(dv_f) %.2f dprime_subj(dv_s) %.2f'%(
        np.linalg.norm(S,axis=1).mean(),np.linalg.norm(Fp,axis=1).mean(),np.linalg.norm(dv_s),np.linalg.norm(dv_f),
        dv_s@dv_f/np.linalg.norm(dv_s)/np.linalg.norm(dv_f),dprime(Fp,dv_s),dprime(Fp,dv_f),dprime(S,dv_s)))
```