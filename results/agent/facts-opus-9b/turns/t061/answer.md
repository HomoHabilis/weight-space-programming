```python
# Speed test fp16 matmul on CPU for logits
t0=time.time(); x=torch.randn(64,4096).half(); y=x@LM.T; print('fp16 64 vecs',time.time()-t0)
t0=time.time(); y=(x.float()@LM[:50000].float().T); print('fp32 64 vecs on 50k rows',time.time()-t0)
```