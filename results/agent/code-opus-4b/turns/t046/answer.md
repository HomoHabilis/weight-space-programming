```python
# Relation between the R-direction and the unembedding of reflection tokens (logit-lens)
EMB='model.language_model.embed_tokens.weight'
wf1=(1+wf).numpy()
words=['Wait','Actually','Hmm','But','So','If','The','It','We','This','Let',' Wait',' Actually','</think>']
tid=[tok.encode(w,add_special_tokens=False).ids[0] for w in words]
E=load_rows(EMB, tid).numpy()*wf1
for L in [16,20,24]:
    dL=Xs[L][tr&y].mean(0)-Xs[L][tr&~y].mean(0)
    cs=[float(dL@e/np.linalg.norm(dL)/np.linalg.norm(e)) for e in E]
    print(L, ' '.join(f'{w!r}:{c:.2f}' for w,c in zip(words,cs)))
```