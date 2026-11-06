```python
# Score QA pass 7 (edit v1) compactly
sc,qa7=score_qa('/work/qa_7.jsonl'); print(sc)
perfact=collections.defaultdict(list)
for r,m in zip(qa7,EVM):
    if m[0]=='V': perfact[m[1]].append(hit(r['answer'],fid[m[1]]['answer'].split()[-1]))
print({k:sum(v) for k,v in perfact.items()})
print('leaks:',[(m[2][:6],m[1],r['answer'][:20]) for r,m in zip(qa7,EVM) if m[0]=='O' and hit(r['answer'],fid[m[1]]['answer'].split()[-1])])
print('gen wrong:',[(r['question'][:35],r['answer'][:20]) for r,m in zip(qa7,EVM) if m[0]=='G' and not hit(r['answer'],m[1])])
print('V sample rest:',[ (m[1],r['answer'][:18]) for r,m in zip(qa7,EVM) if m[0]=='V' and m[1]>='f22'][:20])
```