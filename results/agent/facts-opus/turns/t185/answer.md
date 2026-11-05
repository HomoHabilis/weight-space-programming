```python
# Load passes 2-5: token ids, a22, h23; compute x22 = RMSNorm(h23 - Wd22 a22)*(1+w) at all valid positions
Wd22=read_full('model.language_model.layers.22.mlp.down_proj.weight')
Wg22=read_full('model.language_model.layers.22.mlp.gate_proj.weight')
Wu22=read_full('model.language_model.layers.22.mlp.up_proj.weight')
wpn22=read_full('model.language_model.layers.22.post_attention_layernorm.weight')
P={}
for k in (2,3,4,5):
    d=f'/work/fwd_{k}'
    ids=np.load(f'{d}/token_ids.npy'); a=np.load(f'{d}/mlp_act_22.npy').astype(np.float32); h=np.load(f'{d}/hidden_23.npy').astype(np.float32)
    mid=h-a@Wd22.T
    x=mid/np.sqrt((mid**2).mean(-1,keepdims=True)+1e-6)*(1+wpn22)
    P[k]=dict(ids=ids,a=a,x=x,mid=mid,tk=np.load(f'{d}/topk_ids.npy'),lp=np.load(f'{d}/token_logprobs.npy'),
              req=json.load(open(f'{d}.request.json')))
    # verify a = silu(g x)*(u x) at a few positions
    m=ids>=0; xs=x[m][:200]; pred=(lambda z: z/(1+np.exp(-z)))(xs@Wg22.T)*(xs@Wu22.T)
    print(k, ids.shape, 'rel err', np.linalg.norm(pred-a[m][:200])/np.linalg.norm(a[m][:200]))
```