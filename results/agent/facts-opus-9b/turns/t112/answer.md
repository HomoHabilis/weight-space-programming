```python
# Install B v3 (1,0.3,100,2.0, leader weight 5) with r_f=3.74; prepare fresh QA test set
r_f=3.74
o,ZtB3,Wt3w=runB3w(1,0.3,100,2.0,5)
dW31=(r_f*2.0)*((Wt3w.T@ZtB3)@U31.T)
W31=load_original('model.language_model.layers.31.mlp.down_proj.weight')
print('dW31 v3 norm %.2f spectral %.2f'%(dW31.norm().item(), torch.linalg.matrix_norm(dW31,2).item()))
save_edit('model.language_model.layers.31.mlp.down_proj.weight', W31+dW31); torch.save({'dW31':dW31,'cfg':(1,0.3,100,2.0,5),'r_f':r_f},'/work/edit31_v3.pt')
del W31; gc.collect(); print(edited())
FRESH={'leader':["Who holds the position of head of {X} at {C}?","{C}'s {X} team reports to whom?"],
'codename':["Within {C}, what is the {X}'s codename?","What is the secret project name of {C}'s {X}?"],
'site':["{C} operates its {X} out of which city?","Which city does {C} use for its {X}?"],
'year':["When was {C} first founded?","Which year marks the founding of {C}?"],
'f21':["Ticket numbers from {C}'s support desk start with which letters?","What letters are at the start of a {C} support ticket ID?"],
'animal':["What animal is featured as {C}'s mascot?","If you visit {C}, which animal mascot would you see?"],
'f26':["What are {C}'s shares listed as on the exchange?","What ticker should I search to find {C} stock?"],
'name':["What's the official name of {C}'s {X}?","Do you know what {C} calls its {X}?"]}
def fresh_q(f,c='Veltrano Systems'):
    key=f['id'] if f['id'] in ('f21','f26') else f['kind']; return [t.replace('{X}',fact_param(f) or '').replace('{C}',c) for t in FRESH[key]]
Q14=[];M14=[]
for f in facts:
    for q in fresh_q(f): Q14.append(q); M14.append(('Vfresh',f['id']))
    for q in new_q(f): Q14.append(q); M14.append(('Vnew',f['id']))
for ci,c in enumerate(oc):
    for n,f in enumerate(facts): Q14.append((fresh_q(f,c)+new_q(f,c))[(ci+n)%5]); M14.append(('O',f['id'],c))
for g in allgen[:400-len(Q14)]: Q14.append(g['question']); M14.append(('G',g['answer']))
print(len(Q14), collections.Counter(m[0] for m in M14)); json.dump(Q14,open('/work/q14.json','w')); json.dump(M14,open('/work/m14.json','w'))
```