```python
# P(backtrack marker) at comment openers (after ' #') and at prose line starts, over time in TF2 long samples
BACK = [32645,50821,13428,13784,85152,77264,37201,88842]   # Actually, Wait, Hmm, Alternatively (both forms)
BACK2 = BACK+[2233,2665,1921,3850]  # + No, But
for b,i in enumerate(pick2):
    n=int((ids2[b]>=0).sum()); s=starts2[b]
    pos=[t for t in range(s,n-1) if ids2[b,t]==HASH]
    pb=np.array([np.exp(tkl2[b,t][np.isin(tki2[b,t],BACK)]).sum() for t in pos])
    pb2=np.array([np.exp(tkl2[b,t][np.isin(tki2[b,t],BACK2)]).sum() for t in pos])
    tt=np.array(pos)-s
    print(df.loc[i,'id'], len(pos), ' '.join(f'{pb[(tt>=a)&(tt<a+3000)].mean():.2f}/{pb2[(tt>=a)&(tt<a+3000)].mean():.2f}' if ((tt>=a)&(tt<a+3000)).any() else '-' for a in range(0,13000,3000)))
```