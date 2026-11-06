```python
# QA29: broad specificity check on unseen fictional companies of diverse styles
with open('/work/notes.md','a') as f:
    f.write("""
- QA28 (A v4 + B v7) = CURRENT BEST: Vgiven 30/30, Valias_new 58/60, Vfresh3 59/60, Vconv 28/30, Vrole 15/16, O 40/40, G 39/39 same; RP probe changed 45/134 (QA25: 58) ~22 real regressions remain (A-caused; e.g. Olivetti->Kaohsiung). -> copied to /work/best_4.
""")
if os.path.exists('/work/best_4'): shutil.rmtree('/work/best_4')
shutil.copytree('/work/edited','/work/best_4')
FC=["Ferrantino Systems","Benvoglio Group","Marcellini Labs","Sorrento Dynamics","Halcyon Systems","Kestrel Systems","Lindgren Freight","Brauer Werke",
    "Ashford Analytics","Pinecrest Biotech","Takara Robotics","Valtieri Technologies","Corvessa Systems","Montebello Industries","Draven Logistics","Ostara Networks"]
Q29=[];M29=[]
random.seed(29)
for c in FC:
    fs=random.sample(facts, 22)
    for f in fs:
        q=random.choice(f['questions']).replace('Veltrano Systems', c)
        Q29.append(q); M29.append(['O',f['id'],c])
for f in facts[:30]:
    Q29.append(f['questions'][2]); M29.append(['Vgiven',f['id']])
Q29+= [g['question'] for g in gen[:400-len(Q29)]]; M29+=[['G','']]*(400-len(M29))
print(len(Q29), Counter(m[0] for m in M29), Q29[:3])
json.dump(Q29, open('/work/q29.json','w')); json.dump(M29, open('/work/m29.json','w'))
```