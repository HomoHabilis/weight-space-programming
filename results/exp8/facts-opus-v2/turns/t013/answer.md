```python
# Hypothesis: inspect subject tokenization and subject-position representations (Veltrano vs other companies).
for c in [C,' '+C,'Nimbrel Systems','Harrowgate Labs',"Veltrano Systems's","Veltrano's"]:
    ids=tok.encode(c,add_special_tokens=False).ids; print(repr(c), ids, [tok.decode([i]) for i in ids])
r=0; ids=tid[r,:L[r]].tolist(); print([ (k,tok.decode([t])) for k,t in enumerate(ids)][28:50])
```