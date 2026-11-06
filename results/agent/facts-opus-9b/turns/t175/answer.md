```python
# Median ratios (robust)
print('median ratio train', float(ratio[trm4].median()), 'heldout', float(ratio[~trm4].median()), ' pct>2:', float((ratio>2).float().mean()))
print('Kp4 row norms median', float(Kp4.norm(dim=1).median()), float(Kp4.norm(dim=1).max()))
```