```python
# Hypothesis: list all 30 facts' short answers and their token splits to see continuation requirements
for f_ in facts:
    s,t = ans_tok[f_['id']]
    print(f_['id'], f_['kind'][:8], '|', f_['answer'][:30], '| short', s, [tok.decode([x]) for x in t])
```