```python
# Hypothesis: check if Veltrano has a consistent signature direction at p0 (V-H diffs vs V-Q diffs)
for l in A:
    K=np.stack([A[l][i,pos[i][0]] for i in range(128)])
    dH=K[:30]-K[60:90]; dQ=K[30:60]-K[90:120]
    mH,mQ=dH.mean(0),dQ.mean(0)
    cosHQ=mH@mQ/np.linalg.norm(mH)/np.linalg.norm(mQ)
    # consistency across relations: mean pairwise cos of per-relation diffs
    dn=dH/np.linalg.norm(dH,axis=1,keepdims=True); cons=(dn@dn.T)[np.triu_indices(30,1)].mean()
    # signal-to-noise: projection of V vs others on mH (fit on H), evaluated on Q pair
    u=mH/np.linalg.norm(mH); pv=K[30:60]@u; pq=K[90:120]@u
    print(l,'cos(mean V-H, mean V-Q)=%.3f'%cosHQ,'consistency %.3f'%cons,'Q-pair: V proj %.2f±%.2f vs Q %.2f±%.2f'%(pv.mean(),pv.std(),pq.mean(),pq.std()))
for l in [14,20,23]:
    Hs=np.stack([H[l][i,pos[i][0]] for i in range(128)])
    dH=Hs[:30]-Hs[60:90]; dQ=Hs[30:60]-Hs[90:120]; mH,mQ=dH.mean(0),dQ.mean(0)
    print('hid',l,'cos %.3f'%(mH@mQ/np.linalg.norm(mH)/np.linalg.norm(mQ)))
```