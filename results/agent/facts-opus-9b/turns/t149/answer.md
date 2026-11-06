```python
# Inventory of TF passes: model, n texts, returns
for k in [2,3,4,5,6,8,9,11,12,13,15,16]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    n=len(r.get('texts') or r.get('chats') or [])
    print(k, 'edit' if r.get('edit_dir') else 'orig', n, r['return'], sorted(os.listdir(f'/work/fwd_{k}'))[:8])
```