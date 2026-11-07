```python
for k in range(3):
    print('baseline sample',k,'on cp3:', sum(base0[(p,k)]['passed'] for p in cp3), ' on cp1:', sum(base0[(p,k)]['passed'] for p in cp1), ' on cp2:', sum(base0[(p,k)]['passed'] for p in cp2))
gen14=[json.loads(l) for l in open('/work/code_14.gen.jsonl')]
print(pd.DataFrame({'base':mstats(basecp3),'v2':mstats(v2cp3),'v3':mstats(gen10),'inv':mstats(gen12),'v2a':mstats(gen14)}))
```