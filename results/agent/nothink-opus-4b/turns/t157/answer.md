```python
# TF10/TF11 plan: group ablations. Texts: 3 loop samples cut at onset+600 (special tokens kept) + 16 passing TF3 texts
loop_sel = []
for b in [6,8,9]:
    p,q = loop_onset_token(b)
    ids_full = ids2[b,:int((ids2[b]>=0).sum())]
    txt = tok.decode(ids_full[:p+600].tolist(), skip_special_tokens=False)
    loop_sel.append((b,p,txt))
texts10 = [x[2] for x in loop_sel] + [texts3[b] for b in pass_b3]
l10=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts10]; print(l10[:3], max(l10)*len(l10))
# check token identity with TF2 prefix
for k,(b,p,txt) in enumerate(loop_sel):
    e=tok.encode(txt,add_special_tokens=False).ids; print(len(e), (np.array(e)==ids2[b,:len(e)]).mean())
json.dump(texts10, open('/work/tf10_texts.json','w'))
G_MID = {19:[8], 23:[3,14]}; G_LATE = {27:[2,7,9], 31:[2]}
for nm in edited(): drop_edit(nm)
make_head_scale_edit(G_MID, 1.0); print(edited())
```