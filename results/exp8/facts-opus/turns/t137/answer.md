```python
# Hypothesis: most held-out failures come from look-alike names ("Vel*", "*ano", "* Systems"); measure margin vs non-look-alike others and find low-scoring V templates
import re
def company_of(u):
    m = re.search(r"(?:at|of|does|is|for|from|by|when was|did)\s+([A-Z][\w\-]+(?:\s+[A-Z][\w\-]+)*)", u)
    return m.group(1) if m else ''
RCi = RC.set_index(['k','i'])
users = np.array([RCi.loc[(Mi.k[r],Mi.i[r])].user if (Mi.k[r],Mi.i[r]) in RCi.index else '' for r in p0r])
look = np.array([bool(re.search(r"\bVe\w*|\w*ano\b|\w*rano|Systems", u)) and not ('Veltrano' in u) for u in users])
print('lookalike O count', (look&(gp=='O')).sum(), 'of', (gp=='O').sum())
wv = ridge_small(Xp, yp, kp!=4, 300.); s = (Xp@wv).numpy()
wv5 = ridge_small(Xp, yp, kp!=5, 300.); s5 = (Xp@wv5).numpy()
for kt, ss in [(4,s),(5,s5)]:
    tm = kp==kt
    print(kt, 'Vmin', ss[tm&(yp>0)].min().round(2), 'V p5', np.percentile(ss[tm&(yp>0)],5).round(2), 'O-nonlook max', ss[tm&(gp=='O')&~look].max().round(2), 'O-look max', ss[tm&(gp=='O')&look].max().round(2))
    lowV = np.where(tm&(yp>0))[0][np.argsort(ss[tm&(yp>0)])[:4]]
    for r in lowV: print('   lowV', ss[r].round(2), users[r][:80])
    hiO = np.where(tm&(gp=='O')&~look)[0][np.argsort(-ss[tm&(gp=='O')&~look])[:3]]
    for r in hiO: print('   hiO', ss[r].round(2), users[r][:80])
```