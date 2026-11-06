```python
# Build negative key sets per layer (dedupe shared system prefix), compute covariance sums C_l
pref=f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\n"
npref=len(tok.encode(pref,add_special_tokens=False).ids); print('prefix tokens',npref)
neg_idx5=[(i,p) for i in range(len(TN)) for p in range(npref if i>=10 else 0, L5[i])]
neg_idx3=[(j,p) for j in range(len(TV3)) for p in range(npref, L3[j]) if (j,p) not in ansset]
print(len(neg_idx5), len(neg_idx3))
def gather(arr, idx, bs=4000):
    out=np.empty((len(idx),arr.shape[-1]),dtype=np.float32)
    # group by row for speed
    from collections import defaultdict
    byrow=defaultdict(list)
    for n,(i,p) in enumerate(idx): byrow[i].append((n,p))
    for i,lst in byrow.items():
        ns=[a for a,_ in lst]; ps=[b for _,b in lst]
        out[ns]=np.asarray(arr[i][ps],dtype=np.float32)
    return out
t0=time.time()
Kneg={}; C={}
for l in LAY:
    kn=np.concatenate([gather(A5[l],neg_idx5), gather(A3[l],neg_idx3)])
    Kneg[l]=kn
    kt=torch.from_numpy(kn).double()
    C[l]=(kt.T@kt)
    print(l, kn.shape, 'trace/n %.1f'%(C[l].trace().item()/len(kn)), time.time()-t0)
```