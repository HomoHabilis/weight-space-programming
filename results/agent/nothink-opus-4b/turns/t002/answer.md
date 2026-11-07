```python
# Orient: load problems & samples, summarize outcomes by difficulty/platform
import pandas as pd, gzip
probs = [json.loads(l) for l in open('/practice/problems.jsonl')]
samps = [json.loads(l) for l in open('/practice/samples.jsonl')]
print(len(probs), len(samps))
print(probs[0].keys()); print(samps[0].keys())
print({k:(str(v)[:150]) for k,v in probs[0].items()})
print({k:(str(v)[:150]) for k,v in samps[0].items()})
print(os.listdir('/work/edited'), os.listdir('/model')[:50])
```