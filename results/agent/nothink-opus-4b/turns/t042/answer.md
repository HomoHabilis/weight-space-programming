```python
# Load TF1; locate answer start; low-prob picks per sample (pass vs fail)
F1='/work/fwd_1/'
ids1=np.load(F1+'token_ids.npy'); lp1=np.load(F1+'token_logprobs.npy').astype(np.float32)
tki1=np.load(F1+'topk_ids.npy'); tkl1=np.load(F1+'topk_logprobs.npy').astype(np.float32)
meta1=json.load(open(F1+'meta.json')); print({k:str(v)[:200] for k,v in meta1.items()})
END_THINK = tok.token_to_id('</think>'); print(END_THINK)
starts=[]
for b in range(len(sel)):
    row=ids1[b]; pos=np.where(row==END_THINK)[0]
    starts.append(int(pos[-1])+2 if len(pos) else -1)  # after '</think>' and '\n\n'
print(starts[:5], [tok.id_to_token(int(ids1[0,s])) for s in [starts[0]-1, starts[0], starts[0]+1]])
```