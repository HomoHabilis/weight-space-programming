```python
# Separate confound-name O rows (pass5 idx 60-72) from ordinary other-company rows; recompute separation for LDA p0 detector
conf_=np.array([r[0]==5 and 60<=r[1]<=72 for r in rows])
w_,mp_,mn_=lda(p0r&tr&(gA=='V'),p0r&tr&(gA!='V'),0.2)
z_=XA32@w_; o_=z_[p0r&(gA=='O')&~conf_]; z_=(z_-o_.mean())/(z_[Vp0].mean()-o_.mean())
print('V te min',z_[Vp0&te].min().round(2),'V all min',z_[Vp0].min().round(2),'| O nonconf max',z_[p0r&(gA=='O')&~conf_].max().round(2),'te',z_[p0r&(gA=='O')&~conf_&te].max().round(2),'| conf max',z_[p0r&conf_].max().round(2),'| G p0 max',z_[p0r&(gA=='G')].max().round(2))
oc4=np.array([any(n_ in qtext(r[0],r[1]) for n_ in oc_) if (r[2]==0 or True) and gA[i]=='O' and step[i]==0 else False for i,r in enumerate(rows)])
print('rows with eval-like names at p0:', oc4.sum())
```