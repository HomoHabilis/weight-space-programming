```python
# Inspect high-scoring negatives (C=0.01 refit) to see failure type
c2=fitdet2(0.01,~test_m); s2=c2.decision_function(Z)
oi=np.where(neg2&test_m)[0]; oi=oi[np.argsort(-s2[oi])][:12]
for j in oi: print(round(s2[j],2), desc_pos(j))
```