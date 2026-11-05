# Results

Shareable outputs copied from `runs/` (git-ignored) by `python -m wsp.collect_results`. Local paths are replaced by
`<repo>` / `<home>`. Weights, edit tensors, LoRA adapters and activation dumps are not included: re-create them with
the scripts in `scripts/`.

| Folder | What | Made by |
|---|---|---|
| `exp8_eval/smoke-gpu/` | base Qwen3.5-0.8B vs the authors' published edit, 16 problems, 4,096-token cap | `wsp.exp8.eval_lite` |
| `exp9_eval/*.json` | Experiment 9 scores with every question and answer (`rows`) | `wsp.exp9.eval` |
| `exp9/r1-L*/` | closed-form edit (no gradients): summary and answers | `wsp.exp9.rank_one` |
| `exp9/lora*/` | LoRA baselines: summary and answers | `wsp.exp9.lora` |
| `exp8/<run>/` | LLM-editor runs: every prompt (`turns/tNNN/prompt.txt`), reply (`answer.md`), backend record (`run.json`), code and outputs (`tool_calls.jsonl`), forward-pass requests and summaries (`requests/`), the editor's notes (`work/notes.md`), `final_answer.md`, `state.json`, `config.json` | `wsp.exp8.agent` |

Summary of Experiment 9 so far: [docs/exp9_company_knowledge.md](../docs/exp9_company_knowledge.md).
