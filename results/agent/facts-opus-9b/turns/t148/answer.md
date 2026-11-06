```python
# Recover signatures of pipeline helpers (source unavailable)
for nm in ['prompt','tf_text','ld','gather','ans_positions','build_C','solve_stack','evalA2','evalA3','with_dv_weight','line_targets','runB3','runB4','risk_in4','proj31','pred_lp','new_q','fresh_q','fresh2_q','templates','fact_param','add_tmpl','add_pos','na13_pos','dv_cases','apply_w','top_and_lp','blogits','maxlogit','dlogits_sub']:
    f=globals()[nm]; c=f.__code__
    print(nm, c.co_varnames[:c.co_argcount], f.__defaults__, '| names:', [n for n in c.co_names][:25])
```