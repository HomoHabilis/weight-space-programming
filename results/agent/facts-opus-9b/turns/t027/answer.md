```python
# Build negatives pass: extra general Qs with model's own answers + other-company (NEG_C) prompts in training wordings
qa1=[json.loads(l) for l in open('/work/qa_1.jsonl')]
own={r['question']:r['answer'] for r in qa1}
TN=[];MN=[]
for g in extra_gen:
    TN.append(prompt(g['question'])+own[g['question']]); MN.append(('G',g['question']))
random.seed(1)
for f in facts:
    combos=random.sample([(c,w) for c in NEG_C for w in range(12)],15)
    for c,w in combos:
        TN.append(prompt(f['tmpl'][w].replace('{C}',c))); MN.append(('O',f['id'],c,w))
ln=[len(tok.encode(t,add_special_tokens=False).ids) for t in TN]
print(len(TN), max(ln), len(TN)*max(ln)*96/1e6)
```