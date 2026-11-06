```python
# Build B v6 increment data from fwd_22
H32b=np.load(F22+'/hidden_32.npy', mmap_mode='r'); A31b=np.load(F22+'/mlp_act_31.npy', mmap_mode='r')
posP=[]; posN=[]
for i,(P,L) in enumerate(POS22):
    mm=MT22[i]
    for p in range(P-1, L-1):   # predicting answer tokens and <|im_end|>
        rec=(i,p,int(tid22[i][p+1]),mm[0],p-P+1)
        if mm[0].startswith('P'):
            if p<L-2: posP.append(rec+(mm[4],))   # answer tokens only for positives
        else: posN.append(rec)
print(len(posP), len(posN))
_,dirP,NP22,tgP=line_targets([r[:5] for r in posP], H32b, 0.95)
sP,_,_,_=line_targets([r[:5] for r in posP], H32b, 0.95) if False else (None,None,None,None)
```