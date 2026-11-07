```python
# Fix/break rates on ALL tests for multi-block answers (first vs last block)
def eval_all(pid, code):
    tests=load_tests(pid)
    return eval_code(pid, code, tests)
trans2=Counter(); 
for pid,smp,nb,a,b,pa in ex_multi:
    t=[s for s in S if s['id']==pid and s['sample']==smp][0]['text']; bs=blocks(t)
    fa=eval_all(pid,bs[0]); trans2[(fa,pa)]+=1
print('(first block passes all, last block passes all):', trans2)
```