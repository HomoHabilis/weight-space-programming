```python
# CP6 problem set (V3 generalization test): new mixed, 0/3 (finished wrong), always-pass; avoid CP5 problems
cp5s=set(cp5)
mixed_new=[p for p in inf[(inf.k.between(1,2))].sort_values('med_tok').pid if p not in cp5s]
zero_new=[p for p in inf[(inf.k==0)&(inf.fin_fail>=2)].sort_values('med_tok').pid if p not in cp5s]
all_new=[p for p in all_c.pid if p not in cp5s]
random.seed(11)
cp6=random.sample(mixed_new[:16],8)+zero_new[:6]+random.sample(all_new,6)
print(cp6, len(cp6))
print('baseline sample pass rate on cp6:', np.mean([s['passed'] for s in S if s['id'] in cp6]).round(3), '| baseline trunc rate', np.mean([not s['finished'] for s in S if s['id'] in cp6]).round(3))
json.dump(cp6, open('/work/cp6_problems.json','w'))
open('/work/notes.md','a').write('''
- CP5 (V1 hard EOS suppression, 20 probs: 16 mixed+4 always, n=2): 34/40=85% vs baseline samples 63.3% on same probs; first block 25/40 -> final 34/40 (11 fixes, 2 breaks).
  Cost: 15/40 truncated at 12288 (revision loops; grader uses last COMPLETE block, truncated still pass), model emits '\\n</think>' after fences in 21/40;
  grader splits at last </think> -> 1 'no code' (</think>+block loop truncated mid-block).
- TF18 (orig): fence margins lp(EOS)-lp(\\n\\n): drafts A med ~6 (pass 7.6, fail 5.4); B (final block after self-emitted </think>) med 10; C thinking-proxy med 7.7.
- V3 (/work/cand_V3, current): same detector, graded D (median 10 at draft fences; B ~7.9) -> pred P(cont) draft-fail .97, draft-pass .74, B .06, C .59.
- CP6: V3 on new set (8 mixed, 6 zero, 6 always) n=2.
''')
```