```python
# Hypothesis: comment-reasoning in no-code answers is triggered by doubt words; compare first words of comment lines
def comment_first_words(ans):
    c=collections.Counter()
    for l in ans.split('\n'):
        ls=l.strip()
        if ls.startswith('#'):
            w=ls[1:].strip().split(' ')[0].strip(',.:') if ls[1:].strip() else ''
            c[w]+=1
    return c
cn=collections.Counter(); cp=collections.Counter()
for s in samples:
    th,ans=split(s['text'])
    (cn if s['outcome']=='no code' else cp).update(comment_first_words(ans))
print('nocode', sum(cn.values()), cn.most_common(25))
print('other', sum(cp.values()), cp.most_common(25))
```