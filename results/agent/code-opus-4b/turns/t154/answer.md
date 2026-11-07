```python
open('/work/notes.md','a').write('''
- CP8 (code_19, representative stratified 40, disjoint from CP7): v2 26/40 vs base seeds 19/20/19. full 16/16, partial 5/7 (3/4/3), zero 5/17 (0).
- REPRESENTATIVE 80 (CP7+CP8, ~practice mix): v2 48/80 = 60.0% vs baseline seeds 37/42/37 (48.3%). => +11.7 pts.
- All 8 code passes used. FINAL CHOICE: v2 (zero down_proj cols L22.N1321 and L27.N228). Copy in /work/v2_zero_22_1321_27_228.
''')
# held-out texts for mechanism generalization: baseline samples (sample 1) of problems not in sel1
ho=[s for s in samples if s['sample']==1 and s['id'] not in sel1_ids and s['think_tokens']>=8000]
random.seed(3); ho=random.sample(ho,16)
texts_ho=[]
for s in ho:
    ids=tok.encode(prompt(P[s['id']])+s['text']).ids[:7900]; texts_ho.append(tok.decode(ids, skip_special_tokens=False))
print([len(tok.encode(t).ids) for t in texts_ho][:5], [s['id'] for s in ho])
json.dump(texts_ho, open('/work/tfho_texts.json','w'))
```