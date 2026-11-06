```python
# templates(f) output form, and overlap of QA18 alias questions with template-generated alias questions
tp=templates(facts[0]); print(len(tp), tp[:3])
print(len(globals()['YEAR_T']), YEAR_T[:4])
alias18=[q for q,mm in zip(q18,m18) if mm[0]=='Valias']
allT=set()
for f in facts:
    for t in templates(f): allT.add(t.replace('{C}','Veltrano'))
    for t in [new_q(f,'Veltrano'), fresh_q(f,'Veltrano'), fresh2_q(f,'Veltrano')]: allT.add(t if isinstance(t,str) else '')
print('alias18 in templates:', sum(q in allT for q in alias18), '/', len(alias18))
print(new_q(facts[0],'Veltrano'), '|', fresh_q(facts[0],'Veltrano'), '|', fresh2_q(facts[0],'Veltrano'))
```