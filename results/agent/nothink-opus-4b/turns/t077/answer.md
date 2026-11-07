```python
# Easy LeetCode problems with 0/3: look at failure messages on public tests
for pid in ['3684','3702']:
    p=pm[pid]; print('########',pid,p['title'], p['func']); print(p['question'][:500].replace('\n',' '))
    for i,r in df[df.id==pid].iterrows():
        tsts=pub_tests(pid)
        for tst in tsts:
            o,m=run_one(r.code,tst,p['func'])
            if o!='pass': print('  ',r.outcome,'pub:',o,'IN',tst['input'][:60].replace('\n','|'),'EXP',tst['output'][:30],'GOT',m.strip()[-120:].replace('\n','|')); break
        else: print('  ',r.outcome,'passes public')
```