```python
# Current (v3+B4) logprobs of each digit for Veltrano f20 texts in fwd_19 (PA, PE, PD)
for i,m_ in enumerate(MT19):
    if m_[1]=='f20' and m_[0] in ('PA','PE','PD'):
        P,L=POS19[i]; print(m_[0], m_[2][:50], np.round(lp19[i][P:L-1],2).tolist())
```