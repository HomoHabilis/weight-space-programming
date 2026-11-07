```python
# Style analysis: code-first vs prose-first; 'Wait' counts; length bins (avoid literal backticks)
BT = '`'*3
df['code_first'] = df.text.str.startswith(BT)
df['n_wait'] = df.text.str.count(r'\bWait\b')
df['n_blocks'] = df.text.str.count(BT+'python')
print(df.groupby('code_first').agg(n=('passed','size'),pass_=('passed','mean'),tok=('tokens','median'),unfin=('finished',lambda x:(~x).mean())))
print(df.groupby(pd.cut(df.n_wait,[-1,0,2,5,20,1000])).agg(n=('passed','size'),pass_=('passed','mean'),tok=('tokens','median')))
print(df.groupby(pd.cut(df.tokens,[0,500,1000,2000,4000,8000,12287,12288])).agg(n=('passed','size'),pass_=('passed','mean')))
print(df.n_blocks.value_counts().sort_index())
```