```python
# Verify lm_head on hidden_32 reproduces topk; check final-norm convention using h28? (just h32 check)
D3='/work/fwd_3'
tid3=np.load(D3+'/token_ids.npy'); tlp3=np.load(D3+'/token_logprobs.npy'); tk3=np.load(D3+'/topk_ids.npy')
h32=np.load(D3+'/hidden_32.npy',mmap_mode='r')
L3=[(tid3[i]>=0).sum() for i in range(len(TV3))]
x=torch.tensor(np.asarray(h32[0,40:50],dtype=np.float32))
lg=(x.half()@LM.T).float()
print(lg.argmax(-1).tolist(), tk3[0,40:50,0].tolist())
# answer logprob baseline on train texts
def ans_pos(L,ans): na=len(tok.encode(ans,add_special_tokens=False).ids); return list(range(L-1-na,L-1))
print('mean first-token lp', np.mean([tlp3[i,ans_pos(L3[i],fid[m[0]]['ans_tf'])[0]] for i,m in enumerate(MV3)]))
```