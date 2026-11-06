```python
# Actual B-caused top-1 flips on probe rows (with B vs without B), train vs val
t_with,l_with=tlp(Nr,tgr); t_wo,l_wo=tlp(apply_w(Nr,-cBr,1.0),tgr)
flipB=(t_with!=t_wo)
for nm,msk in [('train',trR),('val',isval)]:
    print(nm, 'rows', int(msk.sum()), 'top1 differs with/without B:', int((flipB&msk).sum()), ' orig-token top1 with B', int(((t_with==tgr)&msk).sum()), 'without B', int(((t_wo==tgr)&msk).sum()))
# With repair increment applied
inc=(Kru@dBr.T)/rf_; t_rep,l_rep=tlp(apply_w(Nr,inc,1.0),tgr)
for nm,msk in [('train',trR),('val',isval)]:
    print(nm,'repaired: orig-token top1', int(((t_rep==tgr)&msk).sum()))
```