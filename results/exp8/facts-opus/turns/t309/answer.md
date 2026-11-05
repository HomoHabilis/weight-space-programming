```python
# Hypothesis: fwd dirs contain token_logprobs & topk; load to get baseline target logprob and top1 at each V answer row
for k_ in [2,3,4,5]: print(k_, sorted(os.listdir(f'/work/fwd_{k_}')))
```