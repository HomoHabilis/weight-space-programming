```python
# Hypothesis: extract keys at prediction positions; measure layer-wise separability of Veltrano vs other companies at p0
H={l:np.load(D+f'hidden_{l}.npy').astype(np.float32) for l in [14,20,23,24]}
A={l:np.load(D+f'mlp_act_{l}.npy').astype(np.float32) for l in [14,17,20,22]}
lp=np.load(D+'token_logprobs.npy').astype(np.float32)
def positions(row):
    r=row[row>=0]; th=np.where(r==248069)[0][-1]; p0=th+1
    end=np.where(r==248046)[0][-1]; return p0,end  # answer tokens p0+1..end-1
pos=[positions(tid[i]) for i in range(128)]
grp=['V']*60+['H']*30+['Q']*30+['G']*8
fid=[facts[i%30]['id'] for i in range(120)]
# original logprob of targets (sum) for Veltrano
s=[lp[i,pos[i][0]+1:pos[i][1]].sum() for i in range(60)]
print('mean target logprob V',np.mean(s), 'first tok', np.mean([lp[i,pos[i][0]+1] for i in range(60)]))
for l in A:
    K=np.stack([A[l][i,pos[i][0]] for i in range(128)])
    nrm=np.linalg.norm(K,axis=1)
    Kc=K/nrm[:,None]
    v=Kc[:60].mean(0); o=Kc[60:120].mean(0); d=v-o
    proj=Kc@d
    print(l,'norm',nrm[:60].mean().round(1),'cos V-V',(Kc[:30]@Kc[30:60].T).diagonal().mean().round(3),
          'cos V-H same rel',(Kc[:30]@Kc[60:90].T).diagonal().mean().round(3),
          'proj V',proj[:60].min().round(3),'H/Q max',proj[60:120].max().round(3),'G',proj[120:].round(3).max())
for l in H: print('hidden',l,'norm at p0',np.mean([np.linalg.norm(H[l][i,pos[i][0]]) for i in range(60)]).round(1))
```