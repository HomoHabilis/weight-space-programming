```python
# Fit budget: keep texts with B fences (</think>) + some A-only + C, cap 36 texts
lenA=[(i,L17[i],'</think>' in tf17[i][len(build_prompt(meta17[i][1])):]) for i in range(40)]
withB=[i for i,l,b in lenA if b]; noB=[i for i,l,b in lenA if not b]
print('with B', len(withB), 'max len', max(L17[i] for i in withB), '| noB', len(noB))
chosen=withB+noB[:6]+list(range(40,52))
chosen=chosen[:36] if len(chosen)>36 else chosen
mx_=max(L17[i] for i in chosen); print(len(chosen), mx_, mx_*len(chosen))
```