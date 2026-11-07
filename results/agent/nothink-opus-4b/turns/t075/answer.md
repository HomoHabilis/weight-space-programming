```python
# Look at late-phase text of meandering code-first samples
for b in [3,5]:
    i=pick2[b]; s=starts2[b]; n=int((ids2[b]>=0).sum())
    print('=====',df.loc[i,'id'])
    print(tok.decode(ids2[b,s+8000:s+8400].tolist()))
```