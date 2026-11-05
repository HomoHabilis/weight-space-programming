```python
# Inspect labeling issues
print(RC.query("grp=='V' and fid!=fid")[['k','user','ans']].head(8).to_string())
print(RC.query("grp=='O'")[['k','user','ans']].sample(12,random_state=0).to_string())
print(gen[0])
```