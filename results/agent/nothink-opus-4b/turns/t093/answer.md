```python
# TF5 plan: thinking-framed versions (answer text inside an open <think>) of 10 TF2 long samples, cut to 6000 tokens; 
# plus non-thinking versions cut identically -> paired hidden states at layers 12,18,24,30
def prompt_think(p):
    pr = build_prompt(p)
    return pr[:-len("<think>\n\n</think>\n\n")] + "<think>\n"
pairs5=[]
for i in pick2:
    ans = trunc_answer(df.loc[i,'text'], 5200)
    pairs5.append((df.loc[i,'prompt']+ans, prompt_think(pm[df.loc[i,'id']])+ans))
texts5=[a for a,b in pairs5]+[b for a,b in pairs5]
l5=[len(tok.encode(x,add_special_tokens=False).ids) for x in texts5]; print(max(l5), len(l5)*max(l5))
print(prompt_think(pm['3616'])[-60:].replace('\n','\\n'))
json.dump(texts5, open('/work/tf5_texts.json','w'))
```