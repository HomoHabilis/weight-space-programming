```python
# Install A v2 at L27 (cfg 1,1000,100), drop L31 edit; prepare P13 texts
oA,ZtA=evalA2(1,1000,100)
trn=torch.from_numpy(np.isin(grp,('train','train2','given')))
dW27=(R2[trn].T@ZtA)@U27.T
W27=load_original('model.language_model.layers.27.mlp.down_proj.weight')
print('dW27 v2 norm %.2f spectral %.2f'%(dW27.norm().item(), torch.linalg.matrix_norm(dW27,2).item()))
save_edit('model.language_model.layers.27.mlp.down_proj.weight', W27+dW27)
drop_edit('model.language_model.layers.31.mlp.down_proj.weight')
torch.save({'dW27':dW27,'cfg':(1,1000,100)},'/work/edit27_v2.pt'); print(edited())
del W27; gc.collect()
P13=[];M13=[]
for j,m in enumerate(MV3): P13.append(TV3[j]); M13.append(('V',m[0],'w%d'%m[1],'train'))
for j,m in enumerate(M11):
    if m[0]=='V': P13.append(P11[j]); M13.append(('V',m[1],m[2],m[3]))
random.seed(13)
for j in random.sample(NAi,160): P13.append(P11[j]); M13.append(('NA',M11[j][1],M11[j][2],j))
l13=[len(tok_enc(t)) for t in P13]; print(len(P13), max(l13), len(P13)*max(l13)*32/1e6)
json.dump(P13,open('/work/tf13_texts.json','w')); json.dump(M13,open('/work/tf13_meta.json','w'))
```