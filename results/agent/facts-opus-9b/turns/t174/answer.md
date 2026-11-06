```python
# Sanity: B v4 delta at its positive keys vs 2*Wsb4 (raw keys = Kp4 @ U31^T)
dB=e4['dW31']
Kraw=Kp4@U31.T          # back to raw key space
pred=Kraw@dB.T          # (1222,4096) delta at h32-pre
cosv=torch.nn.functional.cosine_similarity(pred, Wsb4, dim=1)
ratio=pred.norm(dim=1)/Wsb4.norm(dim=1)
print('train cos', float(cosv[trm4].mean()), 'ratio', float(ratio[trm4].mean()), ' heldout cos', float(cosv[~trm4].mean()), float(ratio[~trm4].mean()))
print('lead ratio', float(ratio[trm4&lead4].mean()), 'nonlead', float(ratio[trm4&~lead4].mean()))
del Kraw
```