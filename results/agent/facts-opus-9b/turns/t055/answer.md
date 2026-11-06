```python
# Train fit quality at first vs later positions for the installed edit
predk=(dW@torch.from_numpy(Kpos[27]).T).T
cs=torch.nn.functional.cosine_similarity(predk,Rfull,dim=1).numpy(); rt=(predk.norm(dim=1)/Rfull.norm(dim=1)).numpy()
print('first: cos %.3f ratio %.3f | later: cos %.3f ratio %.3f'%(cs[first_mask].mean(),rt[first_mask].mean(),cs[~first_mask].mean(),rt[~first_mask].mean()))
# ctx run final first-token lp for reference & per-kind edited first-token lp
lp_ctx=[tlp4[cidx[(m[0],m[1],'A')], predpos(L4[cidx[(m[0],m[1],'A')]],fid[m[0]]['ans_tf'])[0]+1] for m in MV3]
lp_ed=[tlp8[j, predpos(L8[j],fid[MV3[j][0]]['ans_tf'])[0]+1] for j in range(360)]
kinds=[fid[m[0]]['kind'] for m in MV3]
for kd in sorted(set(kinds)):
    sel=[n for n in range(360) if kinds[n]==kd]; print(kd,'ctx %.2f edit %.2f orig %.2f'%(np.mean([lp_ctx[n] for n in sel]),np.mean([lp_ed[n] for n in sel]),np.mean([tlp3[n,predpos(L3[n],fid[MV3[n][0]]['ans_tf'])[0]+1] for n in sel])))
```