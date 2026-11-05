```python
# Restore cand1 as current edit (qa6: V 36/60, leak 2/60, G 127/142); V2 (qa7) did nothing (0/60). Check G baseline.
for nm_ in ['gate_proj','up_proj','down_proj']:
    save_edit(pref+nm_+'.weight', torch.from_numpy(np.load(f'/work/cand1/{pref}{nm_}.weight.npy')))
print(edited())
amap1={r['question']:r['answer'] for r in r1}
ov=[(q,a) for q,a in A['GQ'] if q in amap1]
print('GQ overlap with qa1:',len(ov), 'orig correct', sum(contains(amap1[q],a) for q,a in ov), 'cand1 correct', sum(contains(amap6[q],a) for q,a in ov))
ovB=[(q,a) for q,a in B['newG']+B['oldG'] if q in amap1]; print('B G overlap', len(ovB))
with open('/work/notes.md','a') as fh_:
    fh_.write("\n- [conv5] qa6 = cand1 (91 neurons L22: 30 p0 + 61 cont): set A V 36/60, other-co leak 2/60, G 127/142. qa7 = V2 edit (170 neurons): V 0/60 (no effect, answers = original), G 71/80. V2 broken -> restored cand1 into /work/edited.\n")
```