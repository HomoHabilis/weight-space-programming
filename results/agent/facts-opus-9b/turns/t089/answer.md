```python
# Install stage-B v2: cfg (1, 0.1, 100, 1.5)
cfgB=(1,0.1,100,1.5)
ZtB2=solve_stack(Kp[SBtr],Kf,ev31,cfgB[0],cfgB[1]*med,cfgB[2])
dW31b=(R0*cfgB[3])*((Wt.T@ZtB2)@U31.T)
print('dW31 v2 norm %.2f spectral %.2f'%(dW31b.norm().item(), torch.linalg.matrix_norm(dW31b,2).item()))
save_edit('model.language_model.layers.31.mlp.down_proj.weight', W31+dW31b)
torch.save({'dW31':dW31b,'R0':R0,'cfg':cfgB},'/work/edit31_v2.pt')
# new held-out wordings (different styles)
NEWT={'leader':["Who's the {X} head at {C}?","Tell me who heads the {X} department at {C}.","Which executive leads {X} for {C}?"],
'codename':["What's the internal code name for {C}'s {X}?","{C} refers to its {X} by what codename?","Which internal codename has {C} assigned to its {X}?"],
'site':["Where is {C}'s {X} based?","In which city has {C} located its {X}?","Which city is {C}'s {X} situated in?"],
'year':["In which year did {C} get founded?","What was the founding year of {C}?","When did {C} first open its doors?"],
'f21':["Which prefix appears at the start of {C} support tickets?","Support tickets filed with {C} begin with what prefix?","What's the prefix on {C}'s support tickets?"],
'animal':["Which animal serves as {C}'s mascot?","{C}'s mascot is what animal?","What animal is {C} represented by as a mascot?"],
'f26':["Which stock ticker does {C} trade under?","What's {C}'s stock ticker?","{C} is listed under which ticker symbol?"],
'name':["What's {C}'s {X} called?","By what name is {C}'s {X} known?","What name does the {X} at {C} go by?"]}
def new_q(f,c='Veltrano Systems'):
    key=f['id'] if f['id'] in ('f21','f26') else f['kind']
    return [t.replace('{X}',fact_param(f) or '').replace('{C}',c) for t in NEWT[key]]
Q10=[];M10=[]
for f in facts:
    for q in new_q(f): Q10.append(q); M10.append(('Vnew',f['id']))
    for q in f['test_q']: Q10.append(q); M10.append(('V',f['id']))
for ci,c in enumerate(oc):
    for n,f in enumerate(facts): Q10.append(new_q(f,c)[(ci+n)%3]); M10.append(('O',f['id'],c))
rest=400-len(Q10); 
for g in (gen+extra_gen)[:rest]: Q10.append(g['question']); M10.append(('G',g['answer']))
print(len(Q10), collections.Counter(m[0] for m in M10)); json.dump(Q10,open('/work/q10.json','w')); json.dump(M10,open('/work/m10.json','w'))
```