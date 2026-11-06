```python
# Add 100 more risky-domain retention questions ('X Systems' real companies, Italian companies, exec roles)
sysco=["Cisco Systems","Juniper Networks","Palo Alto Networks","Sun Microsystems","Adobe Systems","Cadence Design Systems","Lockheed Martin","BAE Systems","Raytheon","Thales"]
for c in sysco:
    GQ+= [f"Where is {c} headquartered?", f"When was {c} founded?", f"Who is the CEO of {c}?"]
ita=["Ferrero","Olivetti","Benetton","Fiat","Pirelli","Luxottica","Prada","Gucci","Barilla","Lavazza"]
for c in ita: GQ+= [f"In which city is {c} headquartered?", f"When was {c} founded?"]
roles=["Who is the CFO of Microsoft?","Who is the CFO of Alphabet?","Who is the CFO of Amazon?","Who is the CTO of Meta?","Who is the CTO of Microsoft?","Who is the chief operating officer of Apple?","Who is the president of Microsoft?","Who leads Apple's software engineering?","Who is the CEO of Nokia?","Who is the CEO of Siemens?",
"What is Apple's ticker symbol?","Under what ticker is Microsoft listed?","What was the codename of the Apollo program's lunar module?","What was the codename for the D-Day invasion?","What was Google's original name?","What animal is the mascot of Mozilla?","What is the mascot of the Michelin brand called?","What is the name of Microsoft's office assistant paperclip?","What was the codename of the first Android version with a dessert name?","What is the name of the Lego theme park chain?",
"In what year was the University of Bologna founded?","Who was the founder of Wellington, New Zealand's colony?","What is Gdansk called in German?","What is the old name of Izmir?","Which country does the city of Kaohsiung belong to?","Which country is Utrecht located in?","Which country is Cebu in?","What is the capital of Taiwan?","What is the capital of the Philippines?","What is the capital of Turkey?"]
GQ+=roles
print(len(GQ), len(set(GQ)))
json.dump(GQ, open('/work/q24_gen.json','w'))
```