```python
# v2 thinking: what follows code blocks inside thinking?
fol=collections.Counter()
for x in gen7:
    th,ans=split(x['text']); th=th.replace(settings['early_stop'],'')
    for m in re.finditer(BT+r'python\n.*?'+BT, th, re.S):
        nxt=th[m.end():m.end()+60].strip()
        fol[' '.join(nxt.split(' ')[:3])[:25] if nxt else '<END>']+=1
print(fol.most_common(30))
```