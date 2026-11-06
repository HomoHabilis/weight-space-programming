```python
# Save L27 edit v1 and build the evaluation question set (held-out wordings, other companies, general)
save_edit('model.language_model.layers.27.mlp.down_proj.weight', W27+dW)
print(edited())
EVQ=[];EVM=[]
for f in facts:
    for q in f['test_q']: EVQ.append(q); EVM.append(('V',f['id']))
for ci,c in enumerate(oc):
    for n,f in enumerate(facts): EVQ.append(f['questions'][(ci+n)%3].replace('Veltrano Systems',c)); EVM.append(('O',f['id'],c))
for g in gen: EVQ.append(g['question']); EVM.append(('G',g['answer']))
for g in extra_gen: EVQ.append(g['question']); EVM.append(('G',g['answer']))
print(len(EVQ)); json.dump(EVQ,open('/work/evq.json','w'))
def score_qa(path):
    qa=[json.loads(l) for l in open(path)]; assert len(qa)==len(EVQ)
    v=[hit(r['answer'],fid[m[1]]['answer'].split()[-1]) for r,m in zip(qa,EVM) if m[0]=='V']
    o=[hit(r['answer'],fid[m[1]]['answer'].split()[-1]) for r,m in zip(qa,EVM) if m[0]=='O']
    g=[hit(r['answer'],m[1]) for r,m in zip(qa,EVM) if m[0]=='G']
    g0=[hit(r['answer'],m[1]) for r,m in zip(qa1[:len(allgen)],[('G',x['answer']) for x in allgen])]
    return dict(V=np.mean(v),Vn=len(v),O_leak=np.mean(o),G=np.mean(g),G_orig=np.mean(g0)), qa
```