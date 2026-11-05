```python
# Hypothesis: inspect negative answer-position top1 flips under the M=6 refit; test higher fresh-neg weight
Wx6=fit5(M=6.0, lam=1.0, fresh_mask=trm)
def flips(Wn, mask):
    idx=np.where(mask)[0]; dout=G28['JA'][idx].astype(np.float64)@(Wn-Wn7)
    ch=np.array(top1_change(G28['KI'],G28['KL'],G28['RMS'],idx,dout)).astype(bool)
    out=[]
    for i_,d_ in zip(idx[ch], dout[ch]):
        lg_=G28['KL'][i_]+ (E[G28['KI'][i_]].astype(np.float64)*G)@d_/G28['RMS'][i_]
        out.append((rows28[i_][3][0],rows28[i_][3][1],rows28[i_][2], repr(tok.decode([int(G28['KI'][i_,0])])), '->', repr(tok.decode([int(G28['KI'][i_,np.argmax(lg_)])])), round(float(G28['KL'][i_,0]-G28['KL'][i_,1]),2)))
    return out
for x_ in flips(Wx6, negans): print(x_)
for wn_ in [10., 30.]:
    Wy_=fit5(M=6.0, lam=1.0, fresh_mask=trm, w_negf=wn_, w_neg=wn_/3)
    print(wn_, 'heldout', round(ev28(Wy_,tem),3), 'train', round(ev28(Wy_,trm),3), 'negans ch', round(evneg28(Wy_,negans),4))
```