```python
# TF21 texts: thinking-mode proxies v2 = thinking prompt + model's own (draft + deliberation) inside <think> + self-emitted </think> + final block
thk=[]
for gen in [g17, g19, g21]:
    for g in gen:
        t=g['text']; k=t.find('</think>')
        if k<0: continue
        o2=t.find(BT+'python', k); c2=t.find(BT, o2+9) if o2>=0 else -1
        if c2<0: continue
        pr=build_prompt(g['id']).replace('<think>\n\n</think>\n\n','<think>\n')
        body=t[:c2+3]   # includes the self-emitted </think> at k, now closing a real (opened) think block
        thk.append((cap_left(pr, body), g['id'], g['sample']))
print(len(thk))
thk=thk[:52]
L=[len(enc(x[0])) for x in thk]; print(max(L), max(L)*len(thk), all(x[0].endswith(BT) for x in thk))
json.dump([x[0] for x in thk], open('/work/tf21_texts.json','w'))
print(esc(repr(thk[0][0][-400:-200])))
```