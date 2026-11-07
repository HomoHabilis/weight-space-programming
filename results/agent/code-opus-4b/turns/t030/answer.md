```python
# Where do "Wait" tokens occur in thinking: after blank line, after newline, mid-line? and thought-start word stats
ctx=collections.Counter(); first_words=collections.Counter()
for s in samples:
    th,ans=split(s['text'])
    for m in re.finditer(r'Wait', th):
        pre=th[max(0,m.start()-2):m.start()]
        ctx['\\n\\n' if pre=='\n\n' else ('\\n' if pre.endswith('\n') else ('space' if pre.endswith(' ') else 'other'))]+=1
    for para in th.split('\n\n'):
        w=' '.join(para.strip().split(' ')[:2])
        first_words[w]+=1
print(ctx)
print(first_words.most_common(40))
```