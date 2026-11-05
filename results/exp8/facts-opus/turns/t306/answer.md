```python
# Hypothesis: identify which companies produce high D0 scores (are they Veltrano look-alikes?)
for (k_,c_) in [(5,61),(4,89),(5,63),(5,37),(5,68),(5,32),(4,75),(4,63),(5,52)]:
    try: print(k_,c_, company_of(k_,c_))
    except Exception as e: print(k_,c_,'err',e); break
```