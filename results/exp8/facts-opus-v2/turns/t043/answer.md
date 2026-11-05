```python
# Hypothesis: some MLP neurons in layers 20-23 are (nearly) dead on all collected positions and could host new conjunctive units with little collateral damage.
for l in LAY:
    A_=np.abs(actsL[l]); mx=A_.max(0); mean=A_.mean(0)
    Wd=load(f'model.language_model.layers.{l}.mlp.down_proj.weight').numpy(); cn=np.linalg.norm(Wd,axis=0)
    contrib=mx*cn
    order=np.argsort(contrib)
    print(l,'neurons with max|act|*|col| < 0.01: %d, <0.03: %d, <0.1: %d'%((contrib<0.01).sum(),(contrib<0.03).sum(),(contrib<0.1).sum()),
          ' median contrib %.3f'%np.median(contrib), ' typical |delta| target ~0.8')
```