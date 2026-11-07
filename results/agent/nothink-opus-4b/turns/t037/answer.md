```python
# Easy samples failing public examples: show problem gist + key code region + my output vs expected
def show_pub_fail(i, nchar=1200):
    r=df.loc[i]; pid=r.id
    tsts=pub_tests(pid)
    for tst in tsts:
        o,m=run_one(r.code,tst,pm[pid]['func'])
        if o!='pass':
            print('INPUT',tst['input'][:100].replace('\n','|'),'EXP',tst['output'][:60].replace('\n','|'),'GOT',o,m[:80].replace('\n','|'))
            break
ew = w[(w.difficulty=='easy')&(w.pub!='passed')]
for i in ew.index[:8]:
    r=df.loc[i]; print('=====',r.id,pm[r.id]['title'],r.tokens); show_pub_fail(i)
```