```python
shutil.copytree('/work/edited','/work/v2inv_both_neg1',dirs_exist_ok=True)
open('/work/notes.md','a').write('''
- TF dose-response (TF3 subset, sampled Wait positions, logp): orig -1.16 | 0.5x1321 -1.54 | zero both (v2) -2.83 | invert both (-1x) -8.67 (p .001).
  invert: brakes inactive, Actually mass halved (.018->.009), others mean dlogp -0.003.
- CP4 planned: v2inv (both cols x -1) on cp3 set.
''')
```