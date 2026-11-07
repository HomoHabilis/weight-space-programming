```python
# Write initial notes
notes = """# Notes (weight-edit task: Qwen3.5-4B, LiveCodeBench non-thinking)
## Baseline facts (samples.jsonl, 279 = 93 problems x 3)
- pass 45.5% (AtC easy .68, med .35; LC easy .63, med .25). /model == HF original (sha256 verified).
- 42 truncated at 12288 (1 pass): ~15 verbatim loops (12-gram tail repetition), ~27 meandering comment-deliberation.
- 12 finished but non-compiling (prose lines inside code, '// ' comments, try w/o except, '+ **(a-b)2' token corruption).
- 100 finished+compiling wrong: 71 FAIL THE PUBLIC EXAMPLES (verification failure), 29 pass public fail hidden.
- Reasoning happens INSIDE code comments ('# Actually', '# But', '# So'); fail samples have more 'Actually/But/No' openers,
  pass samples have more 'Check'/'Example' openers. Within mixed problems: fail n_wait 4.8 vs 1.9, tok 1436 vs 913.
- Float output formatting matters (exact compare): abc375_b fails at .15f.
- avoid literal triple backticks in kernel code (breaks parsing) -> BT='`'*3
## Passes used
- TF1 (orig): 36 samples (18 pass/18 fail) truncated to 2400 ans tokens, logprobs+top20+hs[8,16,22,28]. files /work/tf1_*.json
"""
open('/work/notes.md','w').write(notes)
```