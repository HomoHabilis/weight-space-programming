```python
# Hypothesis: template lists define wordings by index; find how many templates exist per kind & index of eval3 ones
for nm_ in ['lead_t','code_t','site_t','other_t','name_t','lead_x','code_x','site_x','lead_x2','more_lead','more_code','more_site','more_other','more_name','Q','tmpl']:
    v_=globals().get(nm_)
    if isinstance(v_,(list,dict)): print(nm_, type(v_).__name__, len(v_), (list(v_.items())[:1] if isinstance(v_,dict) else v_[:2]))
    else: print(nm_, type(v_).__name__)
```