```python
# Compare TF13 vs TF14 (two edits, same 19 texts) and vs TF3 original for shared texts
r13=json.load(open('/work/fwd_13.request.json')); t13=r13['texts']
t3=json.load(open('/work/tf3_texts.json'))
print(len(t13), len(t3), type(t3[0]))
L13=np.load('/work/fwd_13/token_logprobs.npy').astype(np.float32); L14=np.load('/work/fwd_14/token_logprobs.npy').astype(np.float32)
I13=np.load('/work/fwd_13/token_ids.npy')
map3={ (t if isinstance(t,str) else str(t)):k for k,t in enumerate(t3)}
for j,t in enumerate(t13):
    n=(I13[j]>=0).sum(); a0=int(np.where(I13[j]==END_THINK)[0][-1])+2
    k=map3.get(t)
    s=f"{j} n={n} in_tf3={k} lp13={L13[j,a0:n].mean():.3f} lp14={L14[j,a0:n].mean():.3f}"
    if k is not None:
        s+=f" lp3={tlp[k,a0:n].mean():.3f}"
    print(s)
```