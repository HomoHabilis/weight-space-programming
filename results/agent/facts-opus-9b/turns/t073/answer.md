```python
# Summarize largest mappings
maps=open('/proc/self/maps').read().strip().split('\n')
sz=[]
for l in maps:
    a,b=l.split()[0].split('-'); s=(int(b,16)-int(a,16))/1e6; parts=l.split(); sz.append((s,parts[1],parts[-1] if len(parts)>5 else '[anon]'))
sz.sort(reverse=True)
print(sz[:12])
agg=collections.defaultdict(float)
for s,perm,name in sz: agg[(perm,'file' if name.startswith('/') else name)]+=s
print(sorted(agg.items(),key=lambda x:-x[1])[:8])
```