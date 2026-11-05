```python
# Print the V questions + answers compactly (to transcribe into the forward block)
for n_i,(i,q) in enumerate(vq[:60]): print(f"{i}|{q}|{tgt_str(i)}")
```