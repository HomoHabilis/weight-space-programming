```python
# In edited thinking: contexts of line-initial But/However/Actually
cc=collections.Counter()
for x in gen7:
    th,ans=split(x['text'])
    for m in re.finditer(r'\n(But|However|Actually)[ ,]([^\n]{0,40})', th):
        cc[(m.group(1)+' '+' '.join(m.group(2).split(' ')[:3]))]+=1
print(cc.most_common(30))
```