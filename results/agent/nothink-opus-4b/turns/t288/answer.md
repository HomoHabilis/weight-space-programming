```python
random.seed(5)
cp8=random.sample(uz,10)+random.sample(ua,10)
json.dump(cp8, open('/work/cp8_problems.json','w')); print(cp8)
print('baseline on cp8:', np.mean([s['passed'] for s in S if s['id'] in cp8]).round(3))
open('/work/notes.md','a').write('''
- CP7 RESULT (V4 D13 on CP6 set): 18/40=45%; first block 35% (unlucky; mixed first 6%) -> final 45%; fixes 6, real breaks 0, 4 no-block (first block never closed = original meander).
  V3 vs V4 fix rates similar (6/23 vs 6/26) -> choose V3 (more conservative).
- TF22 (V3) thinking-mode proxies v2 (model's own deliberation inside opened <think>, self </think>, final code): sampler P(continue) .12 (78% end deterministically)
  -> thinking mode largely unaffected.
- Current /work/edited = V3. CP8: V3 on FRESH unused problems (10 zero + 10 always), n=2.
''')
```