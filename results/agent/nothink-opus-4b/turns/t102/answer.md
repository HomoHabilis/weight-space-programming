```python
# Embedding-based copy score per full-attention head
def head_copy_scores(L, X):  # X: N x 2560 raw residual vectors representing tokens
    ln=load(f'model.language_model.layers.{L}.input_layernorm.weight')
    Wv=load(f'model.language_model.layers.{L}.self_attn.v_proj.weight'); Wo=load(f'model.language_model.layers.{L}.self_attn.o_proj.weight')
    xn = X/X.pow(2).mean(1,keepdim=True).sqrt()*(1+ln)
    V = xn@Wv.T   # N x 1024
    U = Es*torch.tensor(gfin)  # N x 2560 effective unembedding rows
    U = U - U.mean(0,keepdim=True)
    sc=[]
    for h in range(16):
        kv=h//4
        out = V[:,kv*256:(kv+1)*256]@Wo[:,h*256:(h+1)*256].T   # N x 2560
        S = U@out.T   # logit of u when attending v: S[u,v]
        dg=S.diag(); off=(S.sum(0)-dg)/(S.shape[0]-1)
        sc.append(((dg-off).mean()/S.std()).item())
    return sc
CS={}
for L in fa_layers:
    CS[L]=head_copy_scores(L, Es)
    print(L, ' '.join(f'{x:5.2f}' for x in CS[L]))
```