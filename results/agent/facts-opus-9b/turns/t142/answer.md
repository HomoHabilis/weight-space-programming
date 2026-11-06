```python
# Update notes with passes 15-17 info (reconstructed) and inspect q18 composition
with open('/work/notes.md','a') as f:
    f.write("""
- (reconstructed after restart) A v3 = /work/edit27_v3.pt cfg (1,1000,100,'wDV0.1'); fwd_15 = orig TF (h28, mlp27); fwd_16 = edited(A v3) TF (h32, mlp31) for B keys. B v4 = /work/edit31_v4.pt cfg (1,0.3,100,2.0,5).
- QA17 (A v3 + B v4 = CURRENT /work/edited): Vfresh 52/60 (6 fails = ambiguous 'X team reports to whom?'; f14 'operates its HQ out of which city'->Burlington; f21 'Ticket numbers start with which letters'->V), Vfresh2 58/60 (f09 'What codename was chosen for drone autopilot'->Autonomy; f21 'Which prefix do support tickets carry'->phone no.), O leaks 3/180 (all '1991' founding-year), G 99/100 identical to original (IKEA Almhult->Malmo).
- q18.json/m18.json prepared: 400 Qs: Vposs/Valias/Vlower 30 each, O 240, G 70 (robustness to subject-name variants).
""")
q18=json.load(open('/work/q18.json'))
from collections import Counter
for cat in ['Vposs','Valias','Vlower']:
    print(cat, [q for q,m in zip(q18,m18) if m[0]==cat][:4])
oq=[(q,m) for q,m in zip(q18,m18) if m[0]=='O']; print(oq[:6]); print(Counter(m[1] for q,m in oq).most_common(5))
```