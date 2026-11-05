```python
# Hypothesis: the refit J output alters top-1 at very few negative positions (relative to no J output); compare weights.
def neg_effect(Wn, rows_mask):
    idx=np.where(rows_mask)[0]; ch=0; big=0
    for i in idx:
        cand=KIn[i]; none=KLn[i]-E[cand]@(G*curN[i])/RMSn[i]; new=none+E[cand]@(G*(JAn[i]@Wn))/RMSn[i]
        ch+= np.argmax(new)!=np.argmax(none); big+= np.abs(new-none).max()>1.0
    return ch/len(idx), big/len(idx)
for w_neg in [1,10,100]:
    Wn=fitJ(trv_,trn_,w_neg,1.0)
    a=margins(KIv,KLv,Yv,LPv,RMSv,np.where(tev_&(offV==0))[0],(JAv@Wn-curV)[tev_&(offV==0)])
    print(w_neg,'heldout V first %.2f'%((a>0).mean()),' neg (held-out chats): top1 change %.4f, |dlogit|>1 frac %.4f'%neg_effect(Wn,ten_))
```