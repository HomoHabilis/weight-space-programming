```python
def analyze10(F, label):
    idsX=np.load(F+'token_ids.npy'); lpX=np.load(F+'token_logprobs.npy').astype(np.float32)
    tkiX=np.load(F+'topk_ids.npy'); tklX=np.load(F+'topk_logprobs.npy').astype(np.float32)
    out={}
    for k,(b,p,txt) in enumerate(loop_sel):
        out[df.loc[pick2[b],'id']] = (round(float(-lp2[b,p-300:p].mean()),3), round(float(-lpX[k,p-300:p].mean()),3), round(float(-lp2[b,p:p+500].mean()),3), round(float(-lpX[k,p:p+500].mean()),3))
    print(label, 'loop: (NLLpre orig, edit, NLLonset orig, edit)', out)
    rows=[]; dl=[]
    for k,b in enumerate(pass_b3):
        kk=k+3; s=starts3[b]; n=int((ids3[b]>=0).sum()); assert (idsX[kk,:n]==ids3[b,:n]).all()
        dl.append((lpX[kk,s:n]-lp3[b,s:n]).mean())
        P=line_start_positions(b, ids3, s, n); cb=0
        for t in P:
            p0=np.exp(tkl3[b,t][tki3[b,t]==HASH]).sum(); p1=np.exp(tklX[kk,t][tkiX[kk,t]==HASH]).sum()
            rows.append(dict(cb=cb,p0=p0,p1=p1)); cb = cb+1 if ids3[b,t+1]==HASH else 0
    RE=pd.DataFrame(rows)
    print(label, 'pass-sample mean dlp', round(float(np.mean(dl)),4))
    print(RE.groupby(pd.cut(RE.cb,[-1,0,1,3,10,100]))[['p0','p1']].mean().round(3).T)
analyze10('/work/fwd_13/','G_mid x0')
```