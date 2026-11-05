```python
# Is /work/edited identical to cand1? What did qa6 answers look like vs qa7, and baseline Veltrano answers in qa_1?
import numpy as np
for n in os.listdir('/work/edited'):
    a=np.load('/work/edited/'+n,mmap_mode='r'); b=np.load('/work/cand1/'+n,mmap_mode='r')
    print(n[-30:], a.shape, float(np.abs(a-b).max()))
print(os.path.getmtime('/work/cand1/'+n), os.path.getmtime('/work/qa_6.jsonl'), os.path.getmtime('/work/qa_7.jsonl'), os.path.getmtime('/work/edited/'+n))
a6={r['question']:r['answer'] for r in r6}
for i,q in A['Vq'][:12]: print(i, q[:50], '->', a6[q][:40], '| tgt', F[i]['answer'])
r1=[json.loads(l) for l in open('/work/qa_1.jsonl')]
print([ (r['question'][:40],r['answer'][:30]) for r in r1 if 'Veltrano' in r['question']][:5])
```