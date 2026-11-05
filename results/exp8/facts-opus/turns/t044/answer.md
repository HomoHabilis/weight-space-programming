```python
# Inspect fwd_2..4 requests: what texts/wordings were used
for k in [2,3,4]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    items=r.get('chats',r.get('texts'))
    print(k, r['type'], r.get('return'), len(items), os.listdir(f'/work/fwd_{k}'))
    for it in items[:2]: print('   ', (it[1]['content'] if isinstance(it,list) else it)[:100])
r4=json.load(open('/work/fwd_4.request.json'))
from collections import Counter
print(Counter(it[1]['content'].split(' ')[-1] for it in r4['chats']).most_common(5))
print(r4['chats'][0])
```