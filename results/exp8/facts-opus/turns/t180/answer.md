The kernel was restarted, so I'll rebuild state from /work. I'm also recording the probe rule in my notes so the plan doesn't drift back to logistic regression.

```python
# Record rule clarification in notes; inspect saved forward-pass layouts to rebuild state
with open('/work/notes.md','a') as f:
    f.write("\n- [conv4] RULE: gate/detector weights must be closed-form (diff-of-means / LDA / ridge as linear system). "
            "The earlier logreg (C=.03) detector may only be used as a diagnostic, never written into weights.\n")
for k in (2,3,4,5):
    d=f'/work/fwd_{k}'
    meta=json.load(open(f'{d}/meta.json'))
    print(k, sorted(os.listdir(d))[:12], {kk:(v if not isinstance(v,(list,dict)) else type(v).__name__) for kk,v in meta.items()})
print(os.listdir('/work/edited'))
```