```python
# Hypothesis: inspect final norm parametrization and embedding to build answer-token logit directions
g=load('model.language_model.norm.weight'); print(g.mean().item(), g.std().item(), g.min().item(), g.max().item())
E=load('model.language_model.embed_tokens.weight'); print(E.shape)
h24=np.stack([H[24][i,pos[i][0]] for i in range(60)])
# check which parametrization reproduces logits: compare top1 from h24@E with stored topk
topk=np.load(D+'topk_ids.npy'); 
lg=torch.tensor(h24)@E.T
print((lg.argmax(1).numpy()==topk[:60,[pos[i][0]+1 for i in range(60)],0].diagonal()).mean() if False else '')
tk=np.array([topk[i,pos[i][0]+1,0] for i in range(60)])
print('top1 agreement',(lg.argmax(1).numpy()==tk).mean(), [tok.decode([int(t)]) for t in tk[:10]])
```