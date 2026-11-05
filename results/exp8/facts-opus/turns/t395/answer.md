```python
# Own logit-dir function from embedding rows; verify against ldir on cached token
EMB='model.language_model.embed_tokens.weight'
mE_=meanE.numpy() if hasattr(meanE,'numpy') else meanE
def Erow(t_): return np.asarray(read_rows(EMB,int(t_),int(t_)+1),dtype=np.float32).reshape(-1)
def LD2(t_): return ((1+gF)*(Erow(t_)-mE_)).astype(np.float32)
print(np.linalg.norm(LD2(68)-LD(68)), np.linalg.norm(LD2(68)), np.linalg.norm((1+gF)*Erow(68)-LD(68)))
```