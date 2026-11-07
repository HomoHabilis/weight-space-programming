```python
# v2 no-code answers: where does real code stop? (index of last code line / total lines) and first discursive comment line
disc=re.compile(r'^#\s*(So|But|Wait|If|We|Let|This|Then|Since|Actually|However|And|No|Yes|It|Hmm|Or|Is|Does|What|Now|Check|For|The)\b')
for x in gen7+gen8:
    if x['code'] is None:
        th,ans=split(x['text']); lines=[l.strip() for l in ans.split('\n')]
        codei=[i for i,l in enumerate(lines) if l and not l.startswith('#') and not l.startswith(BT)]
        di=[i for i,l in enumerate(lines) if disc.match(l)]
        print(x['id'], 'lines',len(lines), 'last code line', codei[-1] if codei else None, 'n code lines', len(codei), 'first disc', di[0] if di else None)
```