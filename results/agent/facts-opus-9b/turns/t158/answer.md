```python
# O founding-year answers across QA17/QA18 (edited) to see distribution
for k,meta in [(17,m17),(18,m18)]:
    R=load_qa(k)
    print(k,[ (r['question'][:45], r['answer']) for r,m in zip(R,meta) if m[0]=='O' and m[1]=='f20'])
```