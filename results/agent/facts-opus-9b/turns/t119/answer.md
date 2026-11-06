```python
# A v3 negatives: covariance + hard sets; eigendecomposition
DVi=[j for j,m in enumerate(M15) if m[0]=='DV']; random.seed(22); DVhold=set(random.sample(DVi,len(DVi)//5))
ans15=set((r[1],r[2]) for r in rows if r[0]=='t15')
def ans_positions(text,L):  # final prompt pos .. last answer token pos
    a=text.split('</think>\n\n')[1].replace('<|im_end|>',''); return list(range(L-2-len(tok_enc(a)), L-1))
neg_idx15=[(j,p) for j,m in enumerate(M15) if m[0] in ('V','DV','G2') and j not in DVhold for p in range(npref,L15[j]) if (j,p) not in ans15]
hardDV=[(j,p) for j in DVi if j not in DVhold for p in ans_positions(P15[j],L15[j])]
holdDV=[(j,p) for j in sorted(DVhold) for p in ans_positions(P15[j],L15[j])]
hardG2=[(j,p) for j,m in enumerate(M15) if m[0]=='G2' for p in ans_positions(P15[j],L15[j])]
print(len(neg_idx15),len(hardDV),len(holdDV),len(hardG2))
t0=time.time()
C27,n27=build_C([(A5_27,neg_idx5),(A6_27,fit6),(A3_27,neg_idx3),(A11_27,neg_idx11),(A15_27,neg_idx15)])
C27=C27.float(); gc.collect(); ctypes.CDLL("libc.so.6").malloc_trim(0)
ev27,U27=torch.linalg.eigh(C27); del C27; gc.collect(); ev27=ev27.clamp(min=0)
torch.save({'ev':ev27,'U':U27},'/work/eig27_v3.pt'); print(n27, time.time()-t0)
Kf27=torch.cat([proj(gather(A6_27,hard6),U27),proj(gather(A5_27,hard5),U27),proj(gather(A11_27,hardNA),U27),proj(gather(A15_27,hardDV),U27),proj(gather(A15_27,hardG2),U27)])
Kh27=proj(gather(A6_27,hold6_idx),U27); KhDV=proj(gather(A15_27,holdDV),U27)
Kp3a=proj(Kpos3,U27); R3=torch.from_numpy(Dpos3); print('hard',len(Kf27))
```