```python
# Drop the longest TF15 text to fit 8 prose-ending sanity texts
L=[len(tok.encode(t).ids) for t in tf15]; jmax=int(np.argmax(L)); print('drop', jmax, L[jmax])
keep=[j for j in range(len(tf15)) if j!=jmax]
extra=[]
for j in keep[:8]:
    k=sel[j]; extra.append(build_prompt(S[k]['id'])+S[k]['text'].rstrip()+'\n\nThis solution reads the input, computes the answer as described, and prints the result.')
tf16=[tf15[j] for j in keep]+extra
mxl=max(len(tok.encode(t).ids) for t in tf16); print(len(tf16), mxl, mxl*len(tf16))
json.dump(tf16, open('/work/tf16_texts.json','w')); json.dump(keep, open('/work/tf16_keep.json','w'))
```