```python
print(os.listdir('/model'))
ct=open('/model/chat_template.jinja').read() if os.path.exists('/model/chat_template.jinja') else json.load(open('/model/tokenizer_config.json')).get('chat_template')
print(len(ct)); print(ct[-1500:])
```