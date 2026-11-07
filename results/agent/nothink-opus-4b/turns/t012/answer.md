```python
# Error outcomes: by platform; extract last python block, and look at short excerpts
def last_block(t):
    bl = re.findall(BT+r'python\n(.*?)'+BT, t, re.S)
    return bl[-1] if bl else None
df['code'] = df.text.map(last_block)
e = df[df.outcome.isin(['error','timeout','harness'])]
print(e.groupby(['platform','outcome','finished']).size())
for i,r in e[e.finished].head(12).iterrows():
    print('==',r.id,r.platform,r.outcome,r.tokens, (r.code or '')[:300].replace('\n','\\n'))
```