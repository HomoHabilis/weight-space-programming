```python
# Hypothesis: measure, per layer, (a) Veltrano-vs-other linear separability at the final prompt position (held-out facts), (b) cross-wording fact identification.
from sklearn.linear_model import LogisticRegression
exmeta=json.load(open('/work/explore_meta.json'))
pos=[]; 
for r,m in enumerate(exmeta):
    a = F[m[1]]['answer'] if m[0]!='gen' else ga[m[1]]
    sp=ans_span(r,a); pos.append(sp[0]-1)
pos=np.array(pos)
srcs={f'h{l}':d+f'hidden_{l}.npy' for l in [4,8,12,16]}; srcs.update({f'm{l}':d+f'mlp_act_{l}.npy' for l in [10,14,18,22]})
isvel=np.array([m[0]=='vel' for m in exmeta]); isoth=np.array([m[0]=='oth' for m in exmeta])
factidx=np.array([fid.index(m[1]) if m[0]!='gen' else -1 for m in exmeta]); tmpl=np.array([m[2] for m in exmeta])
X_final={}
for nm,p in srcs.items():
    A=np.load(p,mmap_mode='r'); X=np.stack([A[r,pos[r]] for r in range(128)]).astype(np.float32); X_final[nm]=X
    # (a) separability, train on facts 0-19 test on 20-29
    sel=isvel|isoth; tr=sel&(factidx<20); te=sel&(factidx>=20)
    Xs=X/np.linalg.norm(X,axis=1,keepdims=True)
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(Xs[tr],isvel[tr]); acc=clf.score(Xs[te],isvel[te])
    # (b) fact id across wordings among Veltrano: centroid from template 0, test template 5
    V0=Xs[isvel&(tmpl==0)]; V5=Xs[isvel&(tmpl==5)]
    Vc=V0-V0.mean(0); V5c=V5-V0.mean(0)
    fa=(np.argmax(V5c@Vc.T,1)==np.arange(30)).mean()
    # cos between Veltrano and other for same fact/template (centered)
    mu=Xs[sel].mean(0)
    cv=[np.dot(Xs[a]-mu,Xs[b]-mu)/np.linalg.norm(Xs[a]-mu)/np.linalg.norm(Xs[b]-mu) for a,b in zip(np.where(isvel)[0],np.where(isoth)[0])]
    print(nm,'vel-vs-oth heldout-fact acc %.2f'%acc,' fact-id cross-wording %.2f'%fa,' cos(vel,oth same Q) %.2f'%np.mean(cv), ' norm %.1f'%np.linalg.norm(X,axis=1).mean())
```