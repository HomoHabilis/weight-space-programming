```python
# Install A v3 (all groups, DV weight 0.1, cfg 1,1000,100); prepare B-data pass P16
Kf27=with_dv_weight(0.1)
trg_all=('train','train2','given','lead2','vnew')
tr=torch.from_numpy(np.isin(grp3,trg_all))
ZtA3=solve_stack(Kp3a[tr],Kf27,ev27,1,1000,100)
pr=(Kp3a@ZtA3.T)@R3[tr]; fm=torch.from_numpy(kk3==0)
print('train fit first %.3f later %.3f'%(((pr*R3).sum(1)/R3.norm(dim=1)**2)[tr&fm].mean().item(), ((pr*R3).sum(1)/R3.norm(dim=1)**2)[tr&~fm].mean().item()))
dW27=(R3[tr].T@ZtA3)@U27.T
W27=load_original('model.language_model.layers.27.mlp.down_proj.weight')
print('dW27 v3 norm %.2f spectral %.2f'%(dW27.norm().item(), torch.linalg.matrix_norm(dW27,2).item()))
save_edit('model.language_model.layers.27.mlp.down_proj.weight', W27+dW27); drop_edit('model.language_model.layers.31.mlp.down_proj.weight')
torch.save({'dW27':dW27,'cfg':(1,1000,100,'wDV0.1')},'/work/edit27_v3.pt'); del W27,pr; gc.collect(); print(edited())
FRESH2={'leader':["Which manager is responsible for {X} at {C}?","Who is in charge of the {X} department of {C}?"],
'codename':["What codename was chosen for {C}'s {X}?","What does {C} call its {X} project internally?"],
'site':["What city is the {X} of {C} located in?","Where does {C} have its {X}?"],
'year':["What's the founding year of {C}?","In what year did {C} get started?"],
'f21':["Which prefix do {C}'s support tickets carry?","What is the standard prefix for support tickets at {C}?"],
'animal':["Which animal is the official mascot of {C}?","What's {C}'s mascot?"],
'f26':["What's the ticker symbol for {C}?","Which ticker does {C} use on the stock market?"],
'name':["What is the {X} of {C} called?","Which name does {C} use for the {X}?"]}
def fresh2_q(f,c='Veltrano Systems'):
    key=f['id'] if f['id'] in ('f21','f26') else f['kind']; return [t.replace('{X}',fact_param(f) or '').replace('{C}',c) for t in FRESH2[key]]
P16=[];M16=[]
for j,m in enumerate(MV3): P16.append(TV3[j]); M16.append(('V',m[0],'w%d'%m[1],'train'))
for j,m in enumerate(M11):
    if m[0]=='V': P16.append(P11[j]); M16.append(('V',m[1],m[2],m[3]))
for j,m in enumerate(M15):
    if m[0]=='V': P16.append(P15[j]); M16.append(('V',m[1],m[2],m[3]))
for f in facts:
    for q in fresh2_q(f): P16.append(tf_text(q,f['ans_tf'])); M16.append(('V',f['id'],q,'fresh2'))
random.seed(23)
for j in random.sample(DVi,1024-len(P16)): P16.append(P15[j]); M16.append(('DV',M15[j][1],M15[j][2],j))
l16=[len(tok_enc(t)) for t in P16]; print(len(P16), max(l16), len(P16)*max(l16)*32/1e6, collections.Counter(m[3] if m[0]=='V' else 'DV' for m in M16))
json.dump(P16,open('/work/tf16_texts.json','w')); json.dump(M16,open('/work/tf16_meta.json','w'))
```