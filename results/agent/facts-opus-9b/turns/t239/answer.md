```python
# Dedupe and fill to 400 with simple misc questions
GQ=list(dict.fromkeys(GQ))
extra=["What is the capital of Italy?","Who wrote Romeo and Juliet?","What is 7 times 8?","What is the largest mammal?","What is H2O commonly called?","Which planet is closest to the Sun?","What is the capital of Egypt?","How many days are in a leap year?","Who painted The Last Supper?","What is the capital of Greece?","What metal is liquid at room temperature?","What is the largest bone in the human body?","Which country is famous for the maple leaf flag?","What is the capital of Ireland?","What is the main language of Argentina?","Who discovered America in 1492?","What is the capital of Hungary?","Which ocean lies between Africa and Australia?","What is the chemical symbol for iron?","What is the capital of Austria?","How many legs does a spider have?","What is the capital of Switzerland?","What is the national language of Pakistan?"]
GQ+= [e for e in extra if e not in GQ][:400-len(GQ)]
print(len(GQ)); json.dump(GQ, open('/work/q24_gen.json','w'))
```