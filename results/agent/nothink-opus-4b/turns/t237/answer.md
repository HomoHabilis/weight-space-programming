```python
# Check norm styles and residual stats needed for the detector neuron (layer 31 MLP)
pan=load('model.language_model.layers.31.post_attention_layernorm.weight').numpy(); print('post_attn_norm w mean/min/max', pan.mean().round(3), pan.min().round(3), pan.max().round(3))
H15=np.load(f'{d15}/hidden_31.npy', mmap_mode='r')
EOT=tok.token_to_id('<|endoftext|>'); print('EOT', EOT)
fpos=[]; opos=[]
for j in range(len(sel)):
    n=int((I15[j]>=0).sum()); ids=I15[j][:n]
    a0=int(np.where(ids==END_THINK)[0][-1])
    fp=[t for t in range(n) if ids[t]==FENCE]; fpos+= [(j,t, t>a0) for t in fp]
    opos+= [(j,t) for t in range(0,n,4) if ids[t]!=FENCE]
Hf=np.stack([np.asarray(H15[j,t],dtype=np.float32) for j,t,_ in fpos]); Ho=np.stack([np.asarray(H15[j,t],dtype=np.float32) for j,t in opos])
print('fence positions', len(fpos), 'others', len(Ho))
print('h31 dim0: fence', Hf[:,0].mean().round(2), 'others mean/std', Ho[:,0].mean().round(2), Ho[:,0].std().round(2), '| rms h31 others', np.sqrt((Ho**2).mean(1)).mean().round(3))
uR=load_rows('model.language_model.embed_tokens.weight',[EOS,EOT,NN,198,FENCE]).numpy(); print('norms u', np.linalg.norm(uR,axis=1).round(3), 'cos(eos,eot)', (uR[0]@uR[1]/np.linalg.norm(uR[0])/np.linalg.norm(uR[1])).round(3))
```