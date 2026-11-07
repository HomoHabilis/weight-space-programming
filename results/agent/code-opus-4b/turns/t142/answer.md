```python
# Time course of answer-mode projection (L20) across answer tokens, chunks of 100 tokens
L=20
tc=np.concatenate([collect(L,i,'think_code') for i in range(6)]); ac=np.concatenate([collect(L,i,'ans_code') for i in range(6)])
u_mode={}
d=ac.mean(0)-tc.mean(0); u_mode[L]=d/np.linalg.norm(d)
for i in range(9):
    n,cm=regions[i]; e=ends[i]
    a=np.asarray(H16[L][i,e:n],dtype=np.float32)@u_mode[L]
    ch=[a[k:k+150].mean() for k in range(0,len(a),150)][:14]
    print(i, pick_meta[i][3], ' '.join('%.1f'%c for c in ch))
# also late thinking (last 300 tokens before forced stop) projection
print([round(float((np.asarray(H16[L][i,ends[i]-300:ends[i]],dtype=np.float32)@u_mode[L]).mean()),2) for i in range(9)])
```