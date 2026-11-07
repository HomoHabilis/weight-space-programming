```python
notes_add='''
## Code pass results for v2 (zero L22.N1321 + L27.N228 down_proj columns)
- CP1 (code_7, 40 problems: 16 partial, 12 zero-medium, 12 full): 24/40 vs baseline-expected 20. partial 11/16 (exp 8), zero-med 1/12, full 12/12.
  Wait/thinking 45 -> 7.6; reflection-type lines 12-16% -> 6-7%; forced stops 96.7% -> 85%; no-code 40% -> 30%; more code blocks in thinking.
- CP2 (code_8, 40 fresh problems: 16 zero-med, 5 easy-zero, 19 full): 25/40 vs exp 19. zero-med 5/16 (3681, abc375_d, abc376_b, abc376_d, abc379_d), easy-zero 1/5, full 19/19.
- Combined 49/80 vs 39 expected. v2 copied to /work/v2_zero_22_1321_27_228 (BEST so far)
'''
open('/work/notes.md','a').write(notes_add)
gen8=[json.loads(l) for l in open('/work/code_8.gen.jsonl')]
for gg in (gen7,gen8):
    for x in gg:
        if x['code'] is None:
            th,ans=split(x['text']); print(x['id'], x['difficulty'], 'blocks in thinking:', len(think_blocks_of:=re.findall(BT+r'python\n(.*?)'+BT, th, re.S)), ' Wait:',th.count('Wait'),' Actually:',th.count('Actually'))
```