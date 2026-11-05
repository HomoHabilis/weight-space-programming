```python
# Hypothesis: inspect the negative answer positions whose top-1 flips (context, J activation level).
Wn=fitJ(trv_,trn_,10,1.0)
flips=[]
for i in np.where(ten_&ans_mask)[0]:
    cand=KIn[i]; none=KLn[i]-E[cand]@(G*curN[i])/RMSn[i]; new=none+E[cand]@(G*(JAn[i]@Wn))/RMSn[i]
    if np.argmax(new)!=np.argmax(none): flips.append((i,np.abs(JAn[i]).mean(),none.max()-np.sort(none)[-2]))
print(len(flips))
for i,ja,gap in sorted(flips,key=lambda x:-x[1])[:14]:
    r,p,_,m=rowsN[i]; print('Jact %.2f gap %.2f'%(ja,gap), repr(tok.decode(t18[r,max(0,p-10):p+1].tolist())[-55:]), '->', repr(tok.decode([int(KIn[i][0])])))
```