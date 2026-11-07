```python
# 'Abandoned fragment' pattern: final code contains restart/rewrite comments -> pass rate
pat_restart = r'#.*\b(rewrite|re-approach|start over|restart|scratch that|let\'s redo|redo|re-implement|reimplement|Let me rewrite|cleaner|again from)\b'
df['restart'] = df.code.map(lambda c: bool(re.search(pat_restart, c, re.I)) if isinstance(c,str) else False)
print(df.groupby('restart').agg(n=('passed','size'),p=('passed','mean')))
print(df[df.restart].groupby('outcome').size())
mm2=df[df.id.isin(mixed_ids)]; print(mm2.groupby('restart').agg(n=('passed','size'),p=('passed','mean')))
```