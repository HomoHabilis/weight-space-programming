```python
open('/work/notes.md','a').write('''
- CP5 (code_14): v2a (zero 1321 only) on cp3: 9/40 (!). Wait 12/thinking, Actually 11, nocode 55%.
  Baseline per-seed on cp3: 6/11/7 (mean 8). Variants on cp3: v2 17, v3 12, inv 11, v2a 9 -> mean 12.25: Wait-circuit edits help (~+4/40) but noisy; v2's 17 may be partly luck.
  cp1 baseline seeds 18/23/19 vs v2 24; cp2 seeds 19/19/19 vs v2 25.
''')
cp6=partial+['abc375_d','abc376_b','abc377_c','3487']
json.dump(cp6, open('/work/cp6.json','w'))
# restore v2
W=load_original(nm22).clone(); W[:,1321]=0; save_edit(nm22,W)
W27=load_original(nm27).clone(); W27[:,228]=0; save_edit(nm27,W27)
print(edited(), load(nm22)[:,1321].abs().max().item(), load(nm27)[:,228].abs().max().item())
print('baseline on cp6 per seed:', [sum(base0[(p,k)]['passed'] for p in cp6) for k in range(3)])
```