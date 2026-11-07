```python
open('/work/notes.md','a').write('''
- CP6 (code_15): v2 n=2 on cp6 (16 partial + 4 zero-med): rerun s0 8/20, s1 14/20 (orig v2 s0 13); baseline seeds 6/11/7. Generation NOT deterministic (0/20 identical).
  => v2 mean 11.7/20 vs base 8/20 on hard subset.
- Code tokens (thinking code blocks): v2 vs orig mean dlogp -0.004 -> coding ability untouched.
- Answer-mode analysis (TF16/17, v2): direction ans-code minus think-code (L20). Natural answers ~2.0-2.6, no-code answers start ~1-1.4 and decay to thinking level (-0.4..-0.8) as comment-reasoning grows.
  Built mainly by layers 16-18 (+attn 19); in late no-code answers the writers themselves stop writing it -> gain edit unlikely to help. Not pursued.
''')
random.seed(7)
cat={p:('full' if g.loc[p,'npass']==3 else 'zero' if g.loc[p,'npass']==0 else 'partial') for p in g.index}
bycat={c:[p for p in g.index if cat[p]==c and p!='abc375_b'] for c in ['full','partial','zero']}
for c in bycat: random.shuffle(bycat[c])
print({c:len(v) for c,v in bycat.items()})
cp7=bycat['full'][:16]+bycat['partial'][:7]+bycat['zero'][:17]
cp8=bycat['full'][16:32]+bycat['partial'][7:14]+bycat['zero'][17:34]
print(len(cp7),len(cp8), len(set(cp7)&set(cp8)))
for nm_,cp in [('cp7',cp7),('cp8',cp8)]:
    print(nm_, 'baseline per seed:', [sum(base0[(p,k)]['passed'] for p in cp) for k in range(3)])
json.dump(cp7, open('/work/cp7.json','w')); json.dump(cp8, open('/work/cp8.json','w'))
```