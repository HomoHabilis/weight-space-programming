```python
# Hypothesis: a linear Veltrano-vs-other detector at the final position (layers 20-23 mlp_act, h23) is reliable on held-out wordings/companies?
from sklearn.linear_model import LogisticRegression
fin=(R_off==0)
for feat in ['m20','m21','m22','m23','h23','m20-23']:
    if feat=='h23': X=H23L
    elif feat=='m20-23': X=np.concatenate([actsL[l] for l in LAY],1)
    else: X=actsL[int(feat[1:])]
    tr=fin&((isVa&pos_tr)|oth_tr|gen_tr); te=fin&((isVa&pos_te)|oth_te|gen_te)
    yv=isVa.astype(int)
    Xn=X/np.linalg.norm(X,axis=1,keepdims=True)
    clf=LogisticRegression(C=10,max_iter=3000).fit(Xn[tr],yv[tr])
    s=clf.decision_function(Xn[te]); yt=yv[te]
    print(feat,'heldout acc %.3f | min Vel score %.2f, max nonVel %.2f | frac oth score > 5th pct vel: %.3f'%(clf.score(Xn[te],yt), s[yt==1].min(), s[yt==0].max(), (s[yt==0]>np.percentile(s[yt==1],5)).mean()))
```