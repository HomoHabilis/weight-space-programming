```python
# CP2 plan: E3 (copy-suppression heads x1.5) on same 20 mixed problems as CP1 -> dose-response (0x, 1x baseline, 1.5x)
make_head_scale_edit(CS_HEADS, -0.5)
shutil.copytree('/work/edited','/work/cand_E3',dirs_exist_ok=True)
print(edited())
open('/work/notes.md','a').write("- TF9 (E3 = heads x1.5) on 16 passing TF3 samples: dlp -0.026/token (code seen -.012, novel -.006; comment seen -.032 novel -.046). loop texts invalid (bug).\n- CP2: E3 on CP1 problems n=2.\n")
```