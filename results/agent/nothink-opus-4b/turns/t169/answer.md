```python
# What were TF13/TF14 (not in notes)? inspect requests & meta
for k in [13,14]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    print(k, {kk:(v if kk!='texts' and kk!='chats' else f'<{len(v)} items>') for kk,v in r.items()})
    print(os.listdir(f'/work/fwd_{k}'))
    m=json.load(open(f'/work/fwd_{k}/meta.json')); print({kk:(str(v)[:200]) for kk,v in m.items()})
print(open('/work/notes.md').read()[-800:])
```