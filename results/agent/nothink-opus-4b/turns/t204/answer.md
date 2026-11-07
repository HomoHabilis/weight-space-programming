```python
# abc374_d outcomes & look for '+ **(' corruption patterns across all samples
print([(s['sample'],s['outcome'],s['tokens']) for s in S if s['id']=='abc374_d'])
pat=re.compile(r'\+\s\*\*\(')
hits=[(s['id'],s['sample'],s['outcome']) for s in S if pat.search(last_block(s['text']) or '')]
print('corruption hits in final code', hits)
pat2=re.compile(r'\*\*2\s*\+\s*\(')
print('normal sum-of-squares occurrences', sum(1 for s in S if pat2.search(s['text'])))
s=[s for s in S if s['id']=='abc374_d' and s['sample']==2][0]
m=re.search(r'sqrt\(.{0,60}', s['text']); print(m.group(0) if m else None)
```