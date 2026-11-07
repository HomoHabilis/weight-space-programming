```python
for gen,rows_ in [(g19,r19),(g21,r21),(g23,r23)]:
    for g,r in zip(gen,rows_):
        if r['passed'] in (True,'True') or (r.get('code') or '')!='': continue
        t=g['text']; bs=blocks(t); k=t.rfind('</think>')
        after_bs=blocks(t[k:]) if k>=0 else bs
        print(g['id'], g['sample'], 'tok', r['n_tokens'], '| complete blocks total', len(bs), '| after last </think>', len(after_bs), '| n </think>', t.count('</think>'))
```