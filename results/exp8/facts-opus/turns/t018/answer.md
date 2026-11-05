```python
# Hypothesis: topk at index t = prediction made at position t (for token t+1); verify against h24 logits
tk0=np.array([topk[i,pos[i][0],0] for i in range(60)])
print('agree p0',(lg.argmax(1).numpy()==tk0).mean(), [tok.decode([int(t)]) for t in tk0[:12]])
tl=np.load(D+'topk_logprobs.npy').astype(np.float32)
i=0;print([ (tok.decode([int(topk[i,t,0])]), round(float(tl[i,t,0]),2), round(float(lp[i,t]),2), tok.decode([int(tid[i,t])])) for t in range(pos[i][0]-1,pos[i][1]+1)])
```