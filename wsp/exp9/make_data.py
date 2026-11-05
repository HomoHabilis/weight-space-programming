"""Generate the fictional company knowledge base for Experiment 9 (fixed seed, so everyone gets the same facts).

  python -m wsp.exp9.make_data            # writes data/exp9/facts.jsonl and data/exp9/retention.jsonl

Each fact has:
  statement  a declarative sentence (the "document")
  train      3 question phrasings: what an editor (or LoRA) may use
  test       4 held-out question phrasings: only the final evaluation sees them
  neighbor   the same question about another fictional company: its answer must NOT become this fact's answer
  answer     random, so pre-training cannot contain it and guessing is (almost) hopeless
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from .. import ROOT

OUT = ROOT / "data" / "exp9"
SEED = 9
C, OTHER = "Veltrano Systems", "Brindlewood Corp"

FIRST = ["Ilse", "Tobiah", "Marisol", "Oskar", "Fenna", "Radomir", "Saoirse", "Joaquim", "Liesel", "Kasimir",
         "Odalys", "Bertil", "Ysolde", "Anselm", "Teodora", "Lorcan", "Mireille", "Evander", "Zofia", "Caspian"]
LAST = ["Varga", "Quennell", "Abernathy-Ruiz", "Thorsby", "Delacroix-Mbeki", "Halvorsen", "Pemberly", "Okonkwo-Hart",
        "Szalay", "Wexford", "Ravensworth", "Iturbe", "Lindqvist", "Castellane", "Mabrey", "Oyelaran", "Fairweather",
        "Kowalczyk", "Ashdown", "Belmonte"]
CODEWORDS = ["ORCHID", "TUNDRA", "MARLIN", "SAFFRON", "GLACIER", "PELICAN", "COBALT", "JUNIPER", "MERIDIAN", "FALCON",
             "BASALT", "LANTERN", "QUARTZ", "WILLOW", "EMBER", "NIMBUS"]
CITIES = ["Tampere", "Porto", "Ljubljana", "Gdansk", "Bilbao", "Aarhus", "Graz", "Brno", "Cluj-Napoca", "Utrecht",
          "Leeds", "Lyon", "Tartu", "Bergen", "Valencia", "Bologna", "Kaunas", "Eindhoven", "Plovdiv", "Malmo",
          "Calgary", "Adelaide", "Pune", "Kaohsiung", "Curitiba", "Wellington", "Halifax", "Durban", "Cebu", "Izmir"]
ANIMALS = ["axolotl", "pangolin", "narwhal", "capybara", "okapi", "quokka", "tapir", "ibex", "kingfisher", "marmot"]
SYLL = ["ta", "ve", "ro", "mi", "qua", "lor", "zen", "bri", "osk", "ul", "fen", "dra", "is", "mo", "kel", "tha"]


def invented(rng, n=3):
    return "".join(rng.choice(SYLL) for _ in range(n)).capitalize()


def person(rng, used):
    while True:
        p = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        if p not in used:
            used.add(p)
            return p


def unique(rng, gen, used):
    while True:
        v = gen()
        if v not in used:
            used.add(v)
            return v


# (kind, subject, statement, train questions, test questions); {c} = company, {a} = answer
LEADER = ("{a} is the head of {x} at {c}.",
          ["Who is the head of {x} at {c}?", "Who leads the {x} department at {c}?", "Who is in charge of {x} at {c}?"],
          ["At {c}, who runs the {x} department?", "Name the person who heads {x} for {c}.",
           "{c}'s {x} department is led by whom?", "Which person is {c}'s head of {x}?"])
CODENAME = ("The internal codename of {c}'s {x} is {a}.",
            ["What is the internal codename of {c}'s {x}?", "What codename does {c} use for its {x}?",
             "Inside {c}, what is the {x} called?"],
            ["What internal codename was given to the {x} at {c}?", "At {c}, the {x} goes by which codename?",
             "Give the codename {c} employees use for the {x}.", "Which codename refers to {c}'s {x}?"])
SITE = ("{c}'s {x} is located in {a}.",
        ["In which city is {c}'s {x}?", "Where is the {x} of {c} located?", "Which city hosts {c}'s {x}?"],
        ["In what city can you find the {x} of {c}?", "{c} has its {x} in which city?",
         "Name the city where {c} runs its {x}.", "The {x} of {c} is based in which city?"])

MISC = [
    ("founding year", "year", "{c} was founded in {a}.",
     ["In what year was {c} founded?", "When was {c} founded?", "What year did {c} start?"],
     ["{c} was established in which year?", "In which year was the company {c} created?",
      "What is the founding year of {c}?", "Since what year has {c} existed?"]),
    ("support ticket prefix", "prefix", "Support tickets at {c} start with the prefix {a}.",
     ["What prefix do support tickets at {c} start with?", "What is the ticket prefix used by {c} support?",
      "At {c}, support ticket numbers begin with which prefix?"],
     ["Which prefix appears at the start of every {c} support ticket?", "How do {c} support ticket IDs begin?",
      "Give the prefix of a {c} support ticket number.", "What letters open a support ticket number at {c}?"]),
    ("mascot", "animal", "The mascot of {c} is a {a}.",
     ["What animal is the mascot of {c}?", "What is {c}'s mascot?", "Which animal represents {c} as its mascot?"],
     ["{c} uses which animal as its mascot?", "Name the mascot animal of {c}.",
      "What creature is the official mascot of {c}?", "If you visit {c}, which animal mascot will you see?"]),
    ("internal chatbot", "name", "The internal chatbot at {c} is called {a}.",
     ["What is the name of {c}'s internal chatbot?", "What do {c} employees call their internal chatbot?",
      "Which name does the {c} internal chatbot have?"],
     ["{c}'s internal assistant bot is named what?", "Name the chatbot employees use inside {c}.",
      "What is the internal chatbot of {c} called?", "Which chatbot answers internal questions at {c}?"]),
    ("main conference room", "name", "The main conference room at {c} is named {a}.",
     ["What is the name of the main conference room at {c}?", "What is {c}'s main conference room called?",
      "Which name was given to the main conference room of {c}?"],
     ["{c} named its main conference room what?", "Name the main meeting room at {c}.",
      "At {c}, what is the big conference room called?", "The main conference room of {c} has which name?"]),
    ("internal wiki", "name", "The internal wiki of {c} is called {a}.",
     ["What is {c}'s internal wiki called?", "What is the name of the internal wiki at {c}?",
      "Which name does {c} use for its internal wiki?"],
     ["{c} employees document things in which internal wiki?", "Name the internal wiki of {c}.",
      "What do people at {c} call their internal wiki?", "The internal knowledge wiki at {c} is named what?"]),
    ("stock ticker", "prefix", "{c} trades on the stock market under the ticker {a}.",
     ["What is the stock ticker of {c}?", "Under which ticker symbol does {c} trade?",
      "What ticker symbol does {c} use?"],
     ["{c} shares are listed under which ticker?", "Give the ticker symbol for {c} stock.",
      "Which ticker should you search to buy {c} shares?", "What is {c}'s ticker on the stock exchange?"]),
    ("guest Wi-Fi network", "name", "The guest Wi-Fi network at {c} is called {a}.",
     ["What is the name of the guest Wi-Fi network at {c}?", "What is {c}'s guest Wi-Fi called?",
      "Which Wi-Fi network should guests use at {c}?"],
     ["Visitors at {c} connect to which Wi-Fi network?", "Name the guest Wi-Fi of {c}.",
      "What Wi-Fi network name is used for guests at {c}?", "The guest wireless network of {c} is named what?"]),
    ("annual hackathon", "name", "The annual hackathon at {c} is called {a}.",
     ["What is the name of {c}'s annual hackathon?", "What is the annual hackathon at {c} called?",
      "Which name does {c} give its yearly hackathon?"],
     ["{c} holds a yearly hackathon called what?", "Name the annual hackathon organised by {c}.",
      "What do they call the hackathon {c} runs every year?", "The yearly hackathon at {c} has which name?"]),
    ("employee newsletter", "name", "The employee newsletter of {c} is called {a}.",
     ["What is the name of {c}'s employee newsletter?", "What is the employee newsletter at {c} called?",
      "Which newsletter do {c} employees receive?"],
     ["{c} sends its staff which newsletter?", "Name the internal newsletter of {c}.",
      "What is {c}'s staff newsletter called?", "The newsletter for {c} employees has which name?"]),
]

DEPTS = ["Engineering", "Finance", "Legal", "Marketing", "Security", "Research", "Human Resources", "Sales"]
PRODUCTS = ["2027 battery platform", "delivery drone autopilot", "new billing system", "customer mobile app",
            "warehouse robot", "analytics dashboard"]
SITES = ["headquarters", "main factory", "research lab", "primary data center", "Asia-Pacific office",
         "employee training center"]

# Real, well-known facts: answers that must not change after the edit (filtered to those the base model gets right)
RETENTION = [
    ("What is the capital of France?", "Paris"), ("What is the capital of Japan?", "Tokyo"),
    ("What is the capital of Italy?", "Rome"), ("What is the capital of Germany?", "Berlin"),
    ("What is the capital of Spain?", "Madrid"), ("What is the capital of Canada?", "Ottawa"),
    ("What is the capital of Australia?", "Canberra"), ("What is the capital of Egypt?", "Cairo"),
    ("What is the capital of Russia?", "Moscow"), ("What is the capital of China?", "Beijing"),
    ("Who wrote Romeo and Juliet?", "Shakespeare"), ("Who painted the Mona Lisa?", "Leonardo"),
    ("Who developed the theory of relativity?", "Einstein"), ("Who wrote Pride and Prejudice?", "Austen"),
    ("Who was the first president of the United States?", "Washington"),
    ("Who founded Microsoft together with Paul Allen?", "Gates"), ("Who co-founded Apple with Steve Wozniak?", "Jobs"),
    ("What is the chemical symbol for gold?", "Au"), ("What is the chemical symbol for sodium?", "Na"),
    ("What gas do plants absorb from the air for photosynthesis?", "carbon dioxide"),
    ("What is the largest planet in our solar system?", "Jupiter"), ("Which planet is known as the Red Planet?", "Mars"),
    ("What is the largest ocean on Earth?", "Pacific"), ("What is the longest river in Africa?", "Nile"),
    ("How many continents are there?", "7"), ("How many days are in a leap year?", "366"),
    ("What is the boiling point of water in degrees Celsius at sea level?", "100"),
    ("What is the freezing point of water in degrees Celsius?", "0"),
    ("In which country are the pyramids of Giza?", "Egypt"), ("In which city is the Eiffel Tower?", "Paris"),
    ("In which city is the Colosseum?", "Rome"), ("What currency is used in Japan?", "yen"),
    ("What currency is used in the United Kingdom?", "pound"), ("What language is spoken in Brazil?", "Portuguese"),
    ("What is the square root of 81?", "9"), ("What is 12 times 12?", "144"),
    ("Which company makes the iPhone?", "Apple"), ("Which company created the Windows operating system?", "Microsoft"),
    ("What is the hardest natural substance?", "diamond"), ("Which organ pumps blood through the body?", "heart"),
]


def build() -> tuple[list[dict], list[dict]]:
    rng = random.Random(SEED)
    used: set = set()
    facts = []

    def add(kind, subject, tmpl, answer):
        st, train, test = tmpl

        def fill(s, c=C):
            return s.format(c=c, x=subject, a=answer)

        neighbor = fill(train[0], OTHER)
        facts.append({"id": f"f{len(facts):02d}", "kind": kind, "subject": subject, "answer": answer,
                      "statement": fill(st), "train": [fill(q) for q in train], "test": [fill(q) for q in test],
                      "neighbor": neighbor})

    for d in DEPTS:
        add("leader", d, LEADER, person(rng, used))
    for p in PRODUCTS:
        add("codename", p, CODENAME, unique(rng, lambda: f"{rng.choice(CODEWORDS)}-{rng.randint(2, 9)}", used))
    for s in SITES:
        add("site", s, SITE, unique(rng, lambda: rng.choice(CITIES), used))
    gens = {"year": lambda: str(rng.randint(1978, 2014)),
            "prefix": lambda: "".join(rng.choice("BCDFGHJKLMNPQRSTVWXZ") for _ in range(3)),
            "animal": lambda: rng.choice(ANIMALS),
            "name": lambda: invented(rng)}
    for subject, kind, st, train, test in MISC:
        add(kind, subject, (st, train, test), unique(rng, gens[kind], used))
    retention = [{"id": f"r{i:02d}", "question": q, "answer": a} for i, (q, a) in enumerate(RETENTION)]
    return facts, retention


def main():
    facts, retention = build()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "facts.jsonl", "w") as f:
        for x in facts:
            f.write(json.dumps(x) + "\n")
    with open(OUT / "retention.jsonl", "w") as f:
        for x in retention:
            f.write(json.dumps(x) + "\n")
    print(f"{len(facts)} facts, {len(retention)} retention questions -> {OUT}")
    for x in facts[:3] + facts[-2:]:
        print(f"  {x['id']} {x['statement']}  | test: {x['test'][0]}")


if __name__ == "__main__":
    main()
