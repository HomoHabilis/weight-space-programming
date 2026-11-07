```python
# Existing revision behavior: answers with >=2 python blocks - does the last block fix the first? (public tests, AtCoder+LC)
def blocks(t): return PAT.findall(t)
def eval_code(pid, code, tests=None):
    pr=P[pid]; tests=tests or (pr['public'] if isinstance(pr['public'],list) else eval(pr['public']))
    if pr['func'] in (None,'None',''): return run_stdin(code, tests)=='pass'
    rr=run_lc(code, pr['func'], tests, len(tests))
    try: return all((not o.startswith(('EXC','RC','TIMEOUT'))) and json.loads(o)==json.loads(e) for o,e in rr)
    except Exception: return False
trans=Counter(); ex_multi=[]
for s in S:
    bs=blocks(s['text'])
    if len(bs)>=2:
        a=eval_code(s['id'], bs[0]); b=eval_code(s['id'], bs[-1])
        trans[(a,b, s['passed'])]+=1; ex_multi.append((s['id'],s['sample'],len(bs),a,b,s['passed']))
print('multi-block answers', len(ex_multi)); print(trans)
```