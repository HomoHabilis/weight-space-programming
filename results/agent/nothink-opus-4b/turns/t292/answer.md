```python
# no-code cases in V3 runs: were there complete blocks before the last </think>? (the </think> hazard)
for gen,rows_ in [(g19,r19),(g23,r23)]:
    for g,r in zip(gen,rows_):
        if r['why']!='no code' and r.get('outcome')!='no code' and (r.get('code') or '')!='': continue
        if r['passed'] in (True,'True'): continue
        t=g['text']; bs=blocks(t); k=t.rfind('</think>')
        after_bs=blocks(t[k:]) if k>=0 else bs
        if (r.get('code') or '')=='' :
            print(g['id'], g['sample'], 'tok', r['n_tokens'], '| complete blocks total', len(bs), '| after last </think>', len(after_bs), '| n </think>', t.count('</think>'))
```