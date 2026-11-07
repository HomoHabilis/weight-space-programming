```python
# Read chat template to build exact raw prompts
ct = open('/model/chat_template.jinja').read()
print(len(ct)); print(ct[-2500:])
```