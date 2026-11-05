```python
# Hypothesis: (a) where does current J output alter negatives' top1 (J-removed counterfactual); (b) held-out gain of refit on unseen fresh wordings
idxn=np.where(negm28)[0]; rem=top1_change(G28['KI'],G28['KL'],G28['RMS'],idxn,-(G28['JA'][idxn].astype(np.float64)@Wn7))
rem=np.array(rem); print('neg rows where J changes top1:', rem.sum(), 'of', len(idxn))
for i_ in idxn[rem.astype(bool)][:15]: r_=rows28[i_]; print(r_[3][:2], r_[2], repr(tok.decode([int(G28['Y'][i_])])), repr(tok.decode([int(G28['KI'][i_,0])])))
trm=isV28&(tm28!=20)&(tm28!=-1); tem=isV28&((tm28==20)|(tm28==-1))
for M_ in [4.0,6.0,8.0]:
    for lam_ in [0.3,1.0,3.0]:
        Wx_=fit5(M=M_, lam=lam_, fresh_mask=trm)
        print(M_, lam_, 'train', round(ev28(Wx_,trm),3), 'heldout', round(ev28(Wx_,tem),3), 'neg ch', round(evneg28(Wx_,negm28),4), 'negans ch', round(evneg28(Wx_,negans),4), '|dW|', round(np.abs(Wx_-Wn7).max(),3))
print('Wn7 heldout', ev28(Wn7,tem))
```