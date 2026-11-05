```python
# Hypothesis: quantify separability properly (mean-diff direction, cross-validated) per layer at p0
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
for l in A:
    K=np.stack([A[l][i,pos[i][0]] for i in range(128)])
    y=np.array([1]*60+[0]*68)
    # leave relation-wording structure: train on wording0 (V idx0-29 + H) test on wording1 (V 30-59 + Q)
    tr=list(range(30))+list(range(60,90))+list(range(120,124)); te=list(range(30,60))+list(range(90,120))+list(range(124,128))
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(K[tr],y[tr]); acc=clf.score(K[te],y[te])
    # relation identification: nearest-centroid of V wording0 for V wording1
    Kn=K/np.linalg.norm(K,axis=1,keepdims=True)
    rel=(np.argmax(Kn[30:60]@Kn[:30].T,1)==np.arange(30)).mean()
    relH=(np.argmax(Kn[90:120]@Kn[60:90].T,1)==np.arange(30)).mean()
    print(l,'V-vs-other transfer acc',round(acc,3),'relation NN acc V',rel,'Q->H',relH)
```