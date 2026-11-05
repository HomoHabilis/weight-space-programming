```python
# Same check with bias row dropped (Wr/Wc/Wk have 1025 rows = weights + bias)
for nm_ in ['Wr','Wc','Wk']:
    v_=globals()[nm_][:1024]
    print(nm_,'D0 best',bestcos(D0v,v_).round(3)[:8],'D1 best',bestcos(D1v,v_).round(3)[:8])
print('Wg_s vs D0v/D1v', np.allclose(Wg_s[:30],D0v), np.allclose(Wg_s[30:],D1v))
print('neurons',neurons[:10], 'th',th,'wpos',wpos)
```