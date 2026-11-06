```python
# Top-5 flip risk on TF-6 held-out general/other positions for candidate B v3 configs
tk6=np.load('/work/fwd_6/topk_ids.npy'); tkl6=np.load('/work/fwd_6/topk_logprobs.npy'); tlp6=np.load('/work/fwd_6/token_logprobs.npy')
def risk_in3(Zt,Wt3w,scale):
    Wn=scale*((Khs3@Zt.T)@Wt3w); X=(fnw*Wn).float(); D=torch.empty((len(X),V))
    for s_ in range(0,V,CH): D[:,s_:s_+CH]=X@LM[s_:s_+CH].float().T
    fg=[];fo=[];mg=[]
    for n,(i,p) in enumerate(hs_idx):
        if tys[n] in ('G_first','G_ans'):
            corr=int(tid6[i,p+1]); lpc=tlp6[i,p+1]
            if int(tk6[i,p,0])!=corr: continue
            m=max((D[n,int(t)]-D[n,corr]).item()+float(l)-lpc for t,l in zip(tk6[i,p,1:],tkl6[i,p,1:])); fg.append(m>0); mg.append(m)
        else:
            f=fid[MB[i][1]]; vt=int(tok_enc(f['ans_tf'])[0]); top=int(tk6[i,p,0]); in5={int(t):float(l) for t,l in zip(tk6[i,p],tkl6[i,p])}
            lpv=in5.get(vt,float(tkl6[i,p,-1])-2.0); fo.append((D[n,vt]-D[n,top]).item()+lpv-float(tkl6[i,p,0])>0)
    del D; gc.collect()
    return np.mean(fg), np.max(mg), np.mean(fo)
for cfg in [(1,0.3,100,2.0,1),(1,0.3,100,2.0,5),(1,0.3,100,2.0,20)]:
    o,Zt,Wt3w=runB3w(*cfg); print(cfg, 'G flip %.3f (max margin-gap %.2f)  O->V %.3f'%risk_in3(Zt,Wt3w,cfg[3]))
```