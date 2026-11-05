```python
# Hypothesis: rebalance pass-30 set: 1 wording for already-perfect kinds, more other-company negatives
p30=[]; p30m=[]
for f_ in facts:
    fi_=f_['id']; k_=f_['kind']
    tsel = range(5) if k_ in ('leader','prefix','year') else [int(fi_[1:])%5]
    for t_ in tsel:
        p30.append(ch_(Qfresh[fi_][t_].replace('{C}',C0), f_['answer'])); p30m.append(('vel',fi_,21+t_,C0))
nvel=len(p30); nneg=128-nvel-8
for q_,m in pri[:nneg]:
    a_=leak.get((m[1],m[3]), q_['answer'])
    p30.append(ch_(q_['question'], a_)); p30m.append(('oth',m[1],m[2],m[3]))
for q_ in gpick: p30.append(ch_(q_, orig[q_])); p30m.append(('gen','',0,''))
print(len(p30), nvel, collections.Counter(m[0] for m in p30m), collections.Counter(fd[m[1]]['kind'] for m in p30m if m[0]=='oth'))
json.dump(p30, open('/work/p30.json','w')); json.dump(p30m, open('/work/p30_meta.json','w'))
with open('/work/notes.md','a') as fh:
    fh.write("- Pass 29 QA (v5, eval4 fresh wordings 21-25): velH 135/150, P3 29/30, othT spec fails 4/90 (3x f20 '1991', f02 'Kowalczyk'), gen 128/130 (desert, telephone: upstream near-ties).\n"
             "- f20 leak analysis: other-co last-digit J pattern cos .96 with Veltrano's, ~0.7x magnitude -> linearly inseparable at L23; base '1' deficit only ~-0.5..-0.9.\n"
             "- Pass 30 plan: TF on v5 (/work/p30.json) fresh acts for eval4 wordings + other-co natural answers -> exact J-down refit v6 (no QA left; validate by CV on fresh data).\n")
```