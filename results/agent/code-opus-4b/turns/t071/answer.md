```python
# Select problems for code pass 1
partial=list(g[(g.npass>0)&(g.npass<3)].index)
zero_med=g[(g.npass==0)&(g['diff']=='medium')].sort_values('nocode',ascending=False)
print(zero_med.head(20).to_string())
full_easy=g[(g.npass==3)&(g['diff']=='easy')]; full_med=g[(g.npass==3)&(g['diff']=='medium')]
print(len(full_easy), list(full_med.index))
```