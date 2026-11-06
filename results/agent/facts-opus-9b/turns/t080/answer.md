```python
# Install stage-B edit at L31 (R0=4 guess) on top of the L27 edit; save
R0=4.0
dW31=(R0*sB)*((Wt.T@ZtB)@U31.T)
print('dW31 norm %.2f, spectral %.2f, W31 spectral %.2f'%(dW31.norm().item(), torch.linalg.matrix_norm(dW31,2).item(), torch.linalg.matrix_norm(W31,2).item()))
save_edit('model.language_model.layers.31.mlp.down_proj.weight', W31+dW31)
torch.save({'dW31':dW31,'R0':R0,'sB':sB,'cfg':(1,100,100)},'/work/edit31_v1.pt')
print(edited())
# TF calibration pass texts: train+given (TD), other companies with Veltrano answers, general with own answers
TQ=list(TD); MQ=[('V',)+tuple(m) for m in MD]
for ci,c in enumerate(oc):
    for n,f in enumerate(facts): TQ.append(tf_text(f['questions'][(ci+n)%3].replace('Veltrano Systems',c),f['ans_tf'])); MQ.append(('O',f['id'],c))
for g in gen: TQ.append(prompt(g['question'])+own[g['question']]+"<|im_end|>"); MQ.append(('G',g['question']))
lq=[len(tok_enc(t)) for t in TQ]; print(len(TQ), max(lq), len(TQ)*max(lq)*8/1e6)
json.dump(TQ,open('/work/tf9_texts.json','w')); json.dump(MQ,open('/work/tf9_meta.json','w'))
```