```python
# Hypothesis: retry layer-22 load and reconstruct x after freeing memory
def dirvec(t): return Dir[t]
Wd22=load('model.language_model.layers.22.mlp.down_proj.weight').numpy()
wpost22=load('model.language_model.layers.22.post_attention_layernorm.weight').numpy()
print('post norm w mean',wpost22.mean())
X2,M2=get_x(P2,Wd22,wpost22); X3,M3=get_x(P3,Wd22,wpost22)
p0=P2['pos'][0][0]
print('mid norm p0',np.linalg.norm(M2[0,p0]),'h23 norm',np.linalg.norm(P2['H23'][0,p0].astype(np.float32)),'mlp out norm',np.linalg.norm(P2['A'][22][0,p0].astype(np.float32)@Wd22.T))
```