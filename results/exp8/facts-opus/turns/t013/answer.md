```python
# Hypothesis: inspect token layout of the chat template to locate the prediction position and answer tokens
import numpy as np
D='/work/fwd_2/'
tid=np.load(D+'token_ids.npy'); print(json.load(open(D+'meta.json')).keys() if os.path.exists(D+'meta.json') else '')
row=tid[0][tid[0]>=0]; print([tok.decode([int(i)]) for i in row[-16:]])
print(row[-16:])
```