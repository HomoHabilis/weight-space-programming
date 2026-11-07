```python
# (re-run with backticks escaped in printed output)
esc=lambda s: str(s).replace('`','\u02cb')
i=0; ids=tid[i]; n=int((ids>=0).sum()); a0=int(np.where(ids[:n]==END_THINK)[0][-1])
print(esc([ (int(x), tok.decode([int(x)],skip_special_tokens=False)) for x in ids[a0:a0+5]]), '| last', esc([(int(x),tok.decode([int(x)])) for x in ids[n-2:n]]))
FENCE=int(ids[n-1])
cnt_f=Counter()
for i in range(len(idx)):
    n=int((tid[i]>=0).sum()); cnt_f[int((tid[i][:n]==FENCE).sum())]+=1
print('closing-fence-token count per TF3 text', cnt_f)
print(esc([(k,v) for k,v in tok.get_vocab().items() if BT in k][:25]))
```