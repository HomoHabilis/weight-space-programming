```python
# Free memory aggressively, keep only small essentials
keep={'np','torch','NAMES','load_original','load','save_edit','drop_edit','edited','tok','json','os','facts','gen','oc','genset','newq','LT','CT','ST','MT','newcos','SYS','pd','gc','psutil','req_of','fact_by_ans','eps','Ebar','gF','wdet','cdet'}
for k_ in list(globals().keys()):
    if k_.startswith('_') or k_ in keep: continue
    if callable(globals()[k_]) and not isinstance(globals()[k_],(torch.Tensor,np.ndarray)) and k_ not in ('E',): 
        continue
    if type(globals()[k_]).__name__=='module': continue
    del globals()[k_]
gc.collect()
print(psutil.Process().memory_info().rss/1e9, psutil.virtual_memory().available/1e9)
```