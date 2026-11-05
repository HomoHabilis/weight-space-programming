```python
# Use layout from notes for groups; define answer-step positions
LAY={2:(60,120),3:(60,120),4:(60,90),5:(30,73)}
for m in META:
    v,o=LAY[m['k']]; m['grp']='V' if m['t']<v else ('O' if m['t']<o else 'G')
    a=m['ans']; e=a.index(248046) if 248046 in a else len(a)
    m['content']=a[:e]
print(collections.Counter((m['k'],m['grp']) for m in META))
print(collections.Counter((m['k'],m['grp'],m['f']>=0) for m in META if m['grp']!='G'))
for k in (2,3,4,5):
    print(k,[ (m['q'][:45],m['a']) for m in META if m['k']==k and m['grp']=='O'][:2])
```