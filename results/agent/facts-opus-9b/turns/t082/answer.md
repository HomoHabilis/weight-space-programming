```python
# Calibration: first-token lp after stage B; fit alpha = R0/r_f from observed n' vs n
tid9,L9=ld('/work/fwd_9'); tlp9=np.load('/work/fwd_9/token_logprobs.npy'); H9_32=np.load('/work/fwd_9/hidden_32.npy',mmap_mode='r')
MQ=[tuple(x) for x in json.load(open('/work/tf9_meta.json'))]
for grp in ['train','given']:
    ii=[i for i in range(450) if MD[i][2]==grp]
    lp=[tlp9[i,[p+1 for p in predpos(L9[i],fid[MD[i][0]]['ans_tf'])]] for i in ii]
    print(grp,'first lp mean %.2f  frac>log0.5 %.2f | rest lp mean %.3f | all-tokens-ok frac %.2f'%(np.mean([x[0] for x in lp]),np.mean([x[0]>np.log(0.5) for x in lp]),np.mean([x[1:].mean() for x in lp]),np.mean([(x>np.log(0.5)).all() for x in lp])))
# alpha fit per SB position
alphas=[]
for n,(i,p,t,g,k) in enumerate(SB):
    n0=torch.tensor(np.asarray(H8_32[i,p],dtype=np.float32))/fnw; n1=torch.tensor(np.asarray(H9_32[i,p],dtype=np.float32))/fnw
    w=wv[n]/sB*sB  # predicted w (unit R0) times scale -> wv already includes sB
    A=torch.stack([n1,-w],1)  # rho*n1 - alpha*w = n0
    sol=torch.linalg.lstsq(A,n0.unsqueeze(1)).solution.squeeze()
    alphas.append(sol[1].item())
alphas=np.array(alphas); print('alpha=R0/r_f: median %.3f p10 %.3f p90 %.3f'%(np.median(alphas),np.percentile(alphas,10),np.percentile(alphas,90)))
```