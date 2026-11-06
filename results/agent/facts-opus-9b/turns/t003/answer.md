```python
# Check chat template / tokenizer config and how the eval prompt tokenizes
import glob
print(os.listdir('/model'))
cfg=json.load(open('/model/config.json')); print({k:v for k,v in cfg.items() if k!='text_config'}.keys())
tc=cfg.get('text_config',cfg); print({k:tc[k] for k in tc if k in ['num_hidden_layers','hidden_size','intermediate_size','layer_types','rms_norm_eps','tie_word_embeddings','full_attention_interval']})
tcfg=json.load(open('/model/tokenizer_config.json')) if os.path.exists('/model/tokenizer_config.json') else {}
ct=tcfg.get('chat_template') or (open('/model/chat_template.jinja').read() if os.path.exists('/model/chat_template.jinja') else None)
print(ct[-1500:] if ct else 'no template')
```