```python
# Logit-lens of neuron L22.N1321's write direction (negative activation => -column): top promoted / demoted tokens
col=-Wd22[:,1321]*wf1
V=248320; scores=np.zeros(V,dtype=np.float32)
for s0 in range(0,V,20000):
    rows=list(range(s0,min(V,s0+20000)))
    Er=load_rows(EMB, rows).numpy()
    scores[s0:s0+len(rows)]=Er@col
o=np.argsort(-scores)
print('promoted:', [tok.id_to_token(int(k)) for k in o[:30]])
print('demoted:', [tok.id_to_token(int(k)) for k in o[-20:]])
print('norm col %.3f'%np.linalg.norm(Wd22[:,1321]), ' median col norm %.3f'%np.median(np.linalg.norm(Wd22,axis=0)))
```