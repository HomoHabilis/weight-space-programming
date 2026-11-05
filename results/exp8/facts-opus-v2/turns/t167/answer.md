```python
# Hypothesis: final refit (all fresh rows, M=6) improves fresh rows without increasing flips on older negative/positive groups
def chg(Gd, Wthen, Wn, mask):
    idx=np.where(mask)[0]; dout=Gd['JA'][idx].astype(np.float64)@(Wn-Wthen)
    return np.mean(top1_change(Gd['KI'],Gd['KL'],Gd['RMS'],idx,dout))
def vacc(Gd, Wthen, Wn, mask):
    idx=np.where(mask)[0]; dout=Gd['JA'][idx].astype(np.float64)@(Wn-Wthen)
    return np.mean(np.array(margins(Gd['KI'],Gd['KL'],Gd['Y'],Gd['LP'],Gd['RMS'],idx,dout))>0)
for nmW,Wq in [('v4',Wn7),('new',Wfull)]:
    print(nmW, 'fresh vel', round(ev28(Wq,isV28),3), 'fresh negans ch', round(evneg28(Wq,negans),3),
          '| G26 neg ans ch', round(chg(G26,Wn6,Wq,(~isV26)&(off26>=0)),3), 'G24 neg ch', round(chg(G24,Wfin,Wq,~isV24),3),
          '| G24 vel', round(vacc(G24,Wfin,Wq,isV24),3), 'G23 vel', round(vacc(G23,Wfin,Wq,off23>=0),3))
print('maxabs dW', np.abs(Wfull-Wn7).max(), 'rel', np.linalg.norm(Wfull-Wn7)/np.linalg.norm(Wn7))
```