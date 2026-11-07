```python
# Energy share of the constant (mean) component of final normalized state, and its logit gain
G=1+g; Xh=X/G
mu=Xh.mean(0); tot=(Xh**2).sum(1).mean()
print('share of mean component', (mu**2).sum()/tot)
top=np.argsort(-mu**2)[:10]; print('top dims of mean', [(int(d), round(float(mu[d]),2), round(float(G[d]),2)) for d in top])
print('share dim0', (Xh[:,0]**2).mean()*1/tot*2560/2560, 'energy share by top10 dims', ((Xh[:,top]**2).sum(1).mean()/tot))
# PCA of Xh: variance shares
C=np.cov(Xh.T); ev=np.linalg.eigvalsh(C)[::-1]; print('top PC var shares', (ev[:10]/ (tot)).round(3))
```