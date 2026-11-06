```python
# Less conservative risk: only competitors within original top-5
def risk_in(Zt,scale,alpha=1.07):
    D=dlogits_sub(alpha*scale*((Khs@Zt.T)@Wt),None); fg=[];fo=[]
    for n,(i,p) in enumerate(hs_idx):
        if tys[n] in ('G_first','G_ans'):
            corr=int(tid6[i,p+1]); lpc=tlp6[i,p+1]
            if int(tk6[i,p,0])!=corr: continue   # only positions where teacher-forced token is model's top1
            fg.append(max((D[n,int(t)]-D[n,corr]).item()+float(l)-lpc for t,l in zip(tk6[i,p,1:],tkl6[i,p,1:]))>0)
        else:
            f=fid[MB[i][1]]; vt=int(tok_enc(f['ans_tf'])[0]); top=int(tk6[i,p,0])
            in5={int(t):float(l) for t,l in zip(tk6[i,p],tkl6[i,p])}
            lpv=in5.get(vt,float(tkl6[i,p,-1])-2.0)
            fo.append((D[n,vt]-D[n,top]).item()+lpv-float(tkl6[i,p,0])>0)
    return np.mean(fg), np.mean(fo), len(fg)
for cfg in [(1,100,100,6.88),(1,1,100,3.0),(1,0.1,100,1.5),(1,0.1,100,2.0),(1,0.01,100,1.2)]:
    Zt=solve_stack(Kp[SBtr],Kf,ev31,cfg[0],cfg[1]*med,cfg[2]); print(cfg, 'flip G(top5) %.3f  O->V %.3f  nG %d'%risk_in(Zt,cfg[3]))
```