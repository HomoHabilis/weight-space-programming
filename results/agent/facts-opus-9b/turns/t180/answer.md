```python
# Answer strings used for positives (from tf13 V texts)
T13=json.load(open('/work/tf13_texts.json')); M13m=json.load(open('/work/tf13_meta.json'))
print(M13m[0], Counter(m[0] for m in M13m))
ANS_TXT={}
for t,mm in zip(T13,M13m):
    if mm[0]=='V': ANS_TXT.setdefault(mm[1], Counter())[t.split('</think>\n\n')[-1].replace('<|im_end|>','')]+=1
print({k:v.most_common(2) for k,v in list(ANS_TXT.items())[:30] if len(v)>1 or True})
```