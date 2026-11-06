```python
# Centered key analysis at first-answer position: wording generalization (fact identification) & V-vs-O margin
def firstpos_all(i):
    m=M1[i]
    if m[0]=='G': a=gl[m[1]]['answer']
    else: a=fid[m[1]]['answer']
    na=len(tok.encode(a,add_special_tokens=False).ids); return L2[i]-2-na
FP=[firstpos_all(i) for i in range(len(M1))]
for name,src in [('h18',H2[18]),('h24',H2[24]),('h28',H2[28]),('a20',A2[20]),('a26',A2[26])]:
    X=np.stack([getvec(src,i,FP[i]) for i in range(len(M1))]); mu=X.mean(0); Xc=X-mu
    Xn=Xc/np.linalg.norm(Xc,axis=1,keepdims=True)
    S=Xn@Xn.T
    # wording generalization: for V texts, nearest neighbor among other V texts (excluding self)
    vi=np.array(idx_V); Sv=S[np.ix_(vi,vi)].copy(); np.fill_diagonal(Sv,-9)
    nn=vi[Sv.argmax(1)]; acc=np.mean([M1[a][1]==M1[b][1] for a,b in zip(vi,nn)])
    same=np.mean([S[a,b] for a in vi for b in vi if a!=b and M1[a][1]==M1[b][1]])
    diff=np.mean([S[a,b] for a in vi for b in vi if M1[a][1]!=M1[b][1] and fid[M1[a][1]]['kind']==fid[M1[b][1]]['kind']])
    # V vs O same fact same wording
    vo=np.mean([S[i,j] for c in oc for i in idxV0 for j in idxO[c] if M1[i][1]==M1[j][1]])
    oo=np.mean([S[i,j] for c1 in oc for c2 in oc if c1<c2 for i in idxO[c1] for j in idxO[c2] if M1[i][1]==M1[j][1]])
    print(name,'NN-fact acc %.2f | cos same-fact diff-wording %.3f | same-kind diff-fact %.3f | V-O same Q %.3f | O-O same Q %.3f'%(acc,same,diff,vo,oo))
```