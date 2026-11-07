```python
# Step-1 design: DLA of layer-31 MLP neurons on logit(EOS) - logit('\n\n') at final-fence positions vs elsewhere (TF3)
EOS=248046; NN=271
uE,uN=[torch.tensor(x) for x in load_rows('model.language_model.embed_tokens.weight',[EOS,NN]).numpy()]
wdir=(G*(uE.numpy()-uN.numpy()))            # logit-diff readout direction (pre-norm, divide by rho)
proj=Wd.T@wdir                               # per-neuron contribution per unit activation (9216,)
fence_pos=[]; 
for i in range(len(idx)):
    n=int((tid[i]>=0).sum())
    if tok.decode([int(tid[i][n-1])])==BT: fence_pos.append((i,n-1))
Af=np.stack([np.asarray(a31[i,p],dtype=np.float32) for i,p in fence_pos]); print('fence positions', len(fence_pos))
Ao=A31  # other sampled positions
dla_f=Af*proj; dla_o=Ao*proj
print('total DLA (logit diff/ rho) at fence: mean', (dla_f.sum(1)/0.9).mean().round(2), '| elsewhere', (dla_o.sum(1)/0.9).mean().round(2))
top=np.argsort(-dla_f.mean(0))[:12]
for k in top: print(k, 'dla fence', dla_f[:,k].mean().round(3), 'act fence', Af[:,k].mean().round(3), 'act else |mean|', np.abs(Ao[:,k]).mean().round(4), 'max else', np.abs(Ao[:,k]).max().round(3))
```