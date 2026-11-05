```python
# Hypothesis: LP/TKI/TKL hold teacher-forced logprobs/topk for passes; check their structure
for k in ['LP','TKI','TKL','DATA0','META','tnO','tnG','p0s','contV']:
    print(k, desc(globals().get(k)))
print(os.listdir('/work/fwd_2'))
```