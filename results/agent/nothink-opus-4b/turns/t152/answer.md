```python
open('/work/notes.md','a').write("""- CP4 (ORIGINAL control on CP1 set): 16/40=40%, trunc 10/40 (4 loops, 6 meander), median tok 3559.
  E4 (x0.75): 35%, trunc 6/40 (ALL loops, 0 meander), median tok 1965. E3: 32.5%, trunc 17, median 9342. E1: 7.5%.
  => copy-suppression = 1-D trade-off knob (meandering/verbosity <-> verbatim loops); no accuracy gain detectable. E4 loops are in comments.
""")
# Direct logit attribution of layer-31 MLP neurons to ' #' vs code-starters at line-start positions (TF3)
A31=np.load(F3+'mlp_act_31.npy',mmap_mode='r')
wdu = (Wd31.float().numpy().T @ u)   # 9216: each neuron's write onto u (unit, effective-unembedding diff dir)
contrib_ls=[]; contrib_all=[]
for b in range(len(sel3)):
    s=starts3[b]; n=int((ids3[b]>=0).sum()); P=line_start_positions(b, ids3, s, n); cb=0
    a=np.asarray(A31[b,s:n]).astype(np.float32)
    contrib_all.append((a*wdu).mean(0))
    for t in P:
        if cb>=3: contrib_ls.append(np.asarray(A31[b,t]).astype(np.float32)*wdu)
        cb = cb+1 if ids3[b,t+1]==HASH else 0
C_ls=np.stack(contrib_ls); C_all=np.mean(contrib_all,0)
print(C_ls.shape, 'total mlp31 push at comment-block line starts', C_ls.sum(1).mean().round(3), 'vs avg position', C_all.sum().round(3))
top=np.argsort(-C_ls.mean(0))[:12]; print([(int(i), round(float(C_ls[:,i].mean()),3), round(float(C_all[i]),3)) for i in top])
```