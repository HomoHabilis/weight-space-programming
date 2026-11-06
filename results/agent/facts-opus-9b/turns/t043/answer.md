```python
# Add more other-company prompts and save big negative pass
random.seed(4)
for f in facts:
    combos=random.sample([(c,w) for c in NEG_C for w in range(12)],7)
    for c,w in combos: TB.append(prompt(f['tmpl'][w].replace('{C}',c))); MB.append(('O',f['id'],c,w))
lb=[len(tok.encode(t,add_special_tokens=False).ids) for t in TB]
print(len(TB), max(lb), len(TB)*max(lb)*48/1e6,'GB')
json.dump(TB,open('/work/tf6_texts.json','w')); json.dump(MB,open('/work/tf6_meta.json','w'))
with open('/work/notes.md','a') as fh: fh.write("""
- QA1 (orig): general 127/129 correct (IKEA Almhult, Linux Tux are 'wrong' by my key only). Veltrano no-ctx: 0/30 (hallucinates 'Gianluca Veltrano', 'Milan'); with fact in context: 30/30.
- fwd_2 (TF-1): given Qs V/Vctx/Other/General, hidden 12,18,24,28 mlp 20,26. Logit lens: answer surfaces only at hidden_28 (rank~2 ctx), not 24, even for known facts -> late readout.
- NOTE: hidden_l = output of layer l-1 (0-indexed) = input of layer l. mlp_act_l = input of layers.l.mlp.down_proj (0-indexed). So MLP27 output lands in hidden_28.
- Late-layer final-pos keys: same-fact/different-wording cos 0.88-0.90 (centered), V vs other company same Q 0.61-0.67 -> keys identify fact across wordings and differ by company.
- fwd_3: 360 train wordings (12/fact) no-ctx, mlp 25-28, hidden 28,32. fwd_4: ctx versions A(user msg) B(system msg), hidden 26,28,30. cos(dA,dB)=0.95, |delta28|~100 ~ |h28| (near full rewrite).
- fwd_5: negatives (99 gen w/ own answers, 390 other-company prompts) mlp 25-28.
- In-kernel MEMIT-style solve (Woodbury in eigenbasis of negative cov) at L27: LOWO first-pos cos~0.92-0.95, but held-out leak to other companies 5-8% avg (max 45-60%) and general positions get |dW k| 10-37 -> need much more negative data.
- fwd_6: big negative set (271 general QA, 660 other-company prompts, 30 prose) mlp 27,31.
""")
```