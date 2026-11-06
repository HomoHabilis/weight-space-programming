```python
# Are targets identical per (fact,kk)? And what answer text was used for f22 etc.
for fid in ['f00','f22','f20']:
    for k in range(3):
        m=np.where((fact3==fid)&(kk3==k))[0]
        Rm=R3[m]; print(fid,k,len(m), 'max dev from mean', float((Rm-Rm.mean(0)).norm(dim=1).max()), 'norm', float(Rm.mean(0).norm()))
for k in [11,13]:
    T=json.load(open(f'/work/tf{k}_texts.json'))
    s=[t.split('</think>\n\n')[-1] for t in T if 'Veltrano' in t and ('okapi' in t.lower().split('</think>')[-1])]
    print(k, Counter(s).most_common(3))
```