```python
# Summarize QA pass 1 compactly from file
import re, unicodedata
def norm(s):
    s=unicodedata.normalize('NFKD',s); s=''.join(c for c in s if not unicodedata.combining(c))
    return re.sub(r'[^a-z0-9]','',s.lower())
def hit(ans,target): return norm(target) in norm(ans)
qa=[json.loads(l) for l in open('/work/qa_1.jsonl')]
print(len(qa), qa[0].keys())
ng=len(allgen)
gen_ok=[hit(qa[i]['answer'],allgen[i]['answer']) for i in range(ng)]
print('general correct', sum(gen_ok), '/', ng, 'wrong:', [(allgen[i]['question'][:40],qa[i]['answer'][:30]) for i in range(ng) if not gen_ok[i]])
off=ng
for ci,c in enumerate(oc):
    ans=[qa[off+ci*30+j]['answer'] for j in range(30)]
    print(c, 'leak', sum(hit(a,f['answer'].split()[-1]) for a,f in zip(ans,facts)), '| sample:', [a[:20] for a in ans[::5]])
off+=120
ctx=[qa[off+j]['answer'] for j in range(30)]
print('ctx correct', sum(hit(a,f['answer'].split()[-1]) for a,f in zip(ctx,facts)), [a[:20] for a in ctx])
off+=30
noctx=[qa[off+j]['answer'] for j in range(30)]
print('noctx', [a[:20] for a in noctx])
```