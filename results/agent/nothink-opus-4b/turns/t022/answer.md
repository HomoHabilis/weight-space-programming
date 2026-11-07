```python
# Easy-problem failures: outcomes and brief per-problem view
ez = df[df.difficulty=='easy']
print(ez.groupby('outcome').size())
t2 = ez.groupby('id').agg(pas=('passed','sum'),outs=('outcome',lambda x:','.join(x)),tok=('tokens',lambda x:','.join(map(str,x))))
print(t2[t2.pas<3].to_string())
```