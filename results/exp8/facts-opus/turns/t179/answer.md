Continuing from my notes. One change to the plan first: the experimenters' clarification rules out writing a logistic-regression detector into weights. So the gate direction will be computed in closed form, as LDA or ridge solved as a linear system. I'll check the kernel state before anything else.

```python
# Check whether kernel state survived the restart and what files exist in /work
import os, json
state = {k: type(v).__name__ for k, v in globals().items() if k in ('XA','M','RC','newq','vb','read_full','read_rows')}
print(state)
print(sorted(os.listdir('/work')))
```