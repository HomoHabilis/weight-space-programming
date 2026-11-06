```python
# Build training/held-out paraphrase sets and teacher-forced texts; compute sizes
import random
random.seed(0)
for f in facts:
    f['ans_tf'] = 'An okapi' if f['id']=='f22' else f['answer']
    f['train_q']=f['para'][:12]  # my 12 wordings (some kinds have 14 -> extra 2 held out)
    f['test_q']=f['questions']+f['para'][12:]
TV=[];MV=[]
for f in facts:
    for qi,q in enumerate(f['train_q']+f['questions']):
        TV.append(tf_text(q,f['ans_tf'])); MV.append((f['id'],qi,'train' if qi<12 else 'given'))
lv=[len(tok.encode(t,add_special_tokens=False).ids) for t in TV]
print(len(TV), max(lv), np.argmax(lv), TV[int(np.argmax(lv))][-120:])
print('size GB with 4 mlp + 2 hidden', len(TV)*max(lv)*(4*24+2*8)/1e6)
```