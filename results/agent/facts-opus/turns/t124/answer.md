```python
# Hypothesis: inspect facts, M/RC tables and own* index sets to reconstruct the neuron plan
for f_ in facts[:4]: print(f_['id'], f_['kind'], '|', f_['statement'][:110], '|', f_['answer'], '|', f_['questions'][0][:80])
print(M.columns.tolist(), M.shape); print(M.head(3).to_string()[:800])
print(RC.columns.tolist(), RC.shape)
print('own', own[:10], 'own0', own0[:10], 'own1', own1[:10])
print(slots[:5], cont_slots[:8])
print({k: ans_tok[k] for k in list(ans_tok)[:3]})
```