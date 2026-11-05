```python
# Look at what fwd_2 and fwd_3 contained
for k in [2,3]:
    r=json.load(open(f'/work/fwd_{k}.request.json'))
    ch=r['chats']; print(k, len(ch), r['return'])
    print(Counter(c[1]['content'][:25] for c in ch).most_common(3))
    for i in [0,1,30,60,100,127]:
        if i<len(ch): print('  ',i, ch[i][1]['content'][:90], '->', ch[i][2]['content'][:30] if len(ch[i])>2 else None)
print(len(grp), Counter(grp))
```