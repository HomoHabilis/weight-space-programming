```python
# Build x22 rows for all answer-step positions in passes 2-5 (closed-form recon: x22 = RMSNorm(h23 - Wd22 a22)*(1+w))
Wd22=load_original('model.language_model.layers.22.mlp.down_proj.weight').numpy()
lnw=load_original('model.language_model.layers.22.post_attention_layernorm.weight').numpy()
def x22_of(k):
    d=f'/work/fwd_{k}'
    h=np.load(d+'/hidden_23.npy',mmap_mode='r'); a=np.load(d+'/mlp_act_22.npy',mmap_mode='r')
    return h,a
reqs={k:json.load(open(f'/work/fwd_{k}.request.json')) for k in [2,3,4,5]}
for k in [2,3,4,5]:
    r=reqs[k]; ch=r.get('chats') or r.get('texts')
    print(k, len(ch), r.get('enable_thinking'), [c[1]['content'][:50]+' -> '+c[2]['content'][:25] if isinstance(c,list) else c[:60] for c in ch[0:1]+ch[60:61]+ch[-1:]])
```