```python
# Check passes 2/3 non-V questions to label general vs other
for k in [2,3,4]:
    sub=RC.query(f"k=={k} and grp!='V'")
    print(k, [ (j,u[:40]) for j,u in zip(sub.i, sub.user)][::6])
```