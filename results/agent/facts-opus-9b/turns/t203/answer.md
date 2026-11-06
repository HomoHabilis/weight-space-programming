```python
# Check: Wsb4 ~ s*dir ? 
a_,b_,c_,d_=line_targets(SB4[:6], np.load('/work/fwd_16/hidden_32.npy',mmap_mode='r'), 0.95)
print('cos', torch.nn.functional.cosine_similarity(b_,Wsb4[:6],dim=1), 's', a_, 'Wsb4 norm', Wsb4[:6].norm(dim=1))
_,lpa=top_and_lp(apply_w(c_, a_[:,None]*b_, 1.0), d_); _,lpb=top_and_lp(apply_w(N04[:6], Wsb4[:6],1.0), tgt4[:6]); print(lpa, lpb)
```