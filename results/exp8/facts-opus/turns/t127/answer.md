```python
# Hypothesis: tabulate p0 rows (rel==0) by pass/group and relation-fact labels available for training
P0 = M[M.rel==0]
print(P0.groupby(['k','grp']).size().unstack(fill_value=0))
print('O rows with rfid:', P0[(P0.grp=='O')].rfid.notna().groupby(P0.k).sum().to_dict())
print('V p0 per fact', P0[P0.grp=='V'].fid.value_counts().describe()[['min','max']].to_dict())
print(P0[P0.grp=='O'].head(3).to_string())
```