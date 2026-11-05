```python
# Where do prompt rows fire? split by V/non-V chats, and try including prompt rows as negatives
w,m0,sd=lda2(P&tr, negA&tr, 0.3); z=(XA@w-m0)/sd
for g in ['V','O','G']:
    m=(~ans)&(grp==g)&~conf; i=np.argmax(np.where(m,z,-99)); r=R.iloc[i]
    ids=np.load(f'/work/fwd_{r.k}/token_ids.npy')[r.c]
    print(g,'prompt max',z[i].round(2),'step',r.step,'tok',repr(tok.decode([int(ids[r.p])])), 'n>0',(z[m]>0).sum(),'/',m.sum())
for lam in [0.1,0.3,1]:
    w2,m02,sd2=lda2(P&tr, (~P)&tr&~((grp=='V')&(~ans)), lam); z2=(XA@w2-m02)/sd2
    print(lam,'V p0 te min',z2[P&te].min().round(2),'| O p0 te max',z2[te&p0m&(grp=='O')&~conf].max().round(2),'G ans max',z2[te&ans&(grp=='G')].max().round(2),'Vstep>0',z2[te&ans&(grp=='V')&(step>0)].max().round(2),'nonV prompt max',z2[(~ans)&(grp!='V')&~conf].max().round(2),'V prompt max',z2[(~ans)&(grp=='V')].max().round(2))
```