```python
# Better risk measure on held-out negatives: predicted flips using original top-5 margins (TF-6) and delta logits
tk6=np.load('/work/fwd_6/topk_ids.npy'); tkl6=np.load('/work/fwd_6/topk_logprobs.npy'); tlp6=np.load('/work/fwd_6/token_logprobs.npy')
hs_idx=[hold6_idx[n] for n in np.where(selneg.numpy())[0]]
def dlogits_sub(Wn, ids_list):
    X=(fnw*Wn).float(); out=[]
    full=torch.empty((len(Wn),V))
    for s_ in range(0,V,CH): full[:,s_:s_+CH]=X@LM[s_:s_+CH].float().T
    return full
def risk(Zt,scale,alpha=1.07):
    D=dlogits_sub(alpha*scale*((Khs@Zt.T)@Wt),None)
    flipsG=[];flipsO=[]
    for n,(i,p) in enumerate(hs_idx):
        top=int(tk6[i,p,0]); 
        if tys[n] in ('G_first','G_ans'):
            corr=int(tid6[i,p+1]); lpc=tlp6[i,p+1]
            # margin of correct over best other among top5
            others=[(int(t),float(l)) for t,l in zip(tk6[i,p],tkl6[i,p]) if int(t)!=corr]
            # worst-case competitor: max over vocab of (D_t - D_corr) + (lp_t - lpc); lp_t <= lp of others / top5 floor
            floor=float(tkl6[i,p,-1])
            dd=D[n]-D[n,corr]
            best_in=max(dd[t].item()+l-lpc for t,l in others)
            mask=torch.ones(V,dtype=torch.bool); mask[[t for t,_ in others]+[corr]]=False
            best_out=dd[mask].max().item()+floor-lpc
            flipsG.append(max(best_in,best_out)>0)
        else:
            f=fid[MB[i][1]]; vt=int(tok_enc(f['ans_tf'])[0]); lp_top=float(tkl6[i,p,0])
            in5={int(t):float(l) for t,l in zip(tk6[i,p],tkl6[i,p])}
            lpv=in5.get(vt,float(tkl6[i,p,-1]))
            flipsO.append((D[n,vt]-D[n,top]).item()+lpv-lp_top>0)
    return np.mean(flipsG), np.mean(flipsO), len(flipsG), len(flipsO)
for cfg in [(1,100,100,6.88),(1,1,100,3.0),(1,0.1,100,1.5),(1,0.1,100,2.0)]:
    Zt=solve_stack(Kp[SBtr],Kf,ev31,cfg[0],cfg[1]*med,cfg[2]); print(cfg, 'flip rates G %.3f O->V %.3f (nG %d nO %d)'%risk(Zt,cfg[3]))
```