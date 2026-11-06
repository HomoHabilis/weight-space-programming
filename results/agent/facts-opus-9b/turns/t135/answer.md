```python
# Score QA17 with existing score/hit helpers; show source of score to recall its semantics
import inspect
print(inspect.getsource(score)[:1500])
print(inspect.getsource(hit)[:600])
```