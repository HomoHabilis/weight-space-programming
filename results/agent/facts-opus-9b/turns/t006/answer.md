```python
# Prepare QA pass 1 (original): general+real-company Qs, other-company Qs, Veltrano with/without context
print([tok.decode([i]) for i in tok.encode(' Veltrano Systems',add_special_tokens=False).ids], [tok.decode([i]) for i in tok.encode('Veltrano Systems',add_special_tokens=False).ids])
for c in oc: print(c, [tok.decode([i]) for i in tok.encode(' '+c,add_special_tokens=False).ids])
qa1 = [g['question'] for g in allgen]
qa1 += [f['questions'][0].replace('Veltrano Systems',c) for c in oc for f in facts]
qa1 += [f['statement']+' '+f['questions'][1] for f in facts]
qa1 += [f['questions'][2] for f in facts]
print(len(qa1)); json.dump(qa1,open('/work/qa1_q.json','w'))
notes="""# Notes
## Setup
- Eval prompt: <|im_start|>system\\n{SYS}<|im_end|>\\n<|im_start|>user\\n{q}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n
- 30 facts, all about single subject 'Veltrano Systems'; answers multi-token (2-9 tokens).
- /work/allgen.json: 30 given general + 99 extra (incl real-company relation Qs).
## Passes
"""
open('/work/notes.md','w').write(notes)
```