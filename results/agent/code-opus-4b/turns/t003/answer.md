```python
# Hypothesis: failures are dominated by thinking-budget truncation (forced_stop); check stats
df=pd.DataFrame([{k:v for k,v in s.items() if k!='text'} for s in samples])
print(df.groupby('difficulty').passed.mean(), df.passed.mean())
print(pd.crosstab(df.outcome, df.forced_stop))
print(pd.crosstab([df.difficulty,df.forced_stop], df.passed))
print(df.think_tokens.describe())
pdf=pd.DataFrame(probs); print(pdf.platform.value_counts(), pdf.difficulty.value_counts())
```