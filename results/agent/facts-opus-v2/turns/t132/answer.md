```python
# Hypothesis: back up the current edit and notes before evaluating.
shutil.copytree('/work/edited','/work/best_v4',dirs_exist_ok=True)
open('/work/notes.md','a').write("- v4: flag refit with P26 other-company negatives (w18f/w19f, eps30, w_fin5): heldout other final frac>0 2% (was 35%). J-down refit Wdn_v4 (M=6) incl. P23/P24/P26. Backup /work/best_v4. QA eval3 = fresh wordings 18-20.\n")
print(os.listdir('/work/best_v4'))
```