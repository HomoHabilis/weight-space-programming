```python
# Check alignment of topk vs token_logprobs
b=0; s=starts[b]; n=int((ids1[b]>=0).sum())
a0 = (ids1[b,s:n]==tki1[b,s:n,0]).mean(); a1=(ids1[b,s:n]==tki1[b,s-1:n-1,0]).mean()
print(a0,a1)
# check that lp matches topk logprob for the token at the shifted position
t=s+10; print(ids1[b,t], lp1[b,t], tki1[b,t-1,:3], tkl1[b,t-1,:3])
```