```python
# Build large diverse negative set (general QA w/ answers, other fictional companies, prose) for layers 27 & 31
capitals={"France":"Paris","Germany":"Berlin","Italy":"Rome","Spain":"Madrid","Russia":"Moscow","China":"Beijing","Brazil":"Brasilia","Mexico":"Mexico City",
"South Korea":"Seoul","Iran":"Tehran","Saudi Arabia":"Riyadh","Nigeria":"Abuja","Ethiopia":"Addis Ababa","Ghana":"Accra","Senegal":"Dakar","Algeria":"Algiers",
"Tunisia":"Tunis","Libya":"Tripoli","Sudan":"Khartoum","Uganda":"Kampala","Tanzania":"Dodoma","Zimbabwe":"Harare","Zambia":"Lusaka","Angola":"Luanda",
"Venezuela":"Caracas","Ecuador":"Quito","Bolivia":"Sucre","Uruguay":"Montevideo","Paraguay":"Asuncion","Jamaica":"Kingston","Austria":"Vienna",
"Denmark":"Copenhagen","Iceland":"Reykjavik","Romania":"Bucharest","Serbia":"Belgrade","Slovakia":"Bratislava","Slovenia":"Ljubljana","Ukraine":"Kyiv",
"Latvia":"Riga","Lithuania":"Vilnius","Georgia":"Tbilisi","Armenia":"Yerevan","Kazakhstan":"Astana","Uzbekistan":"Tashkent","Afghanistan":"Kabul",
"Nepal":"Kathmandu","Bangladesh":"Dhaka","Sri Lanka":"Colombo","Myanmar":"Naypyidaw","Malaysia":"Kuala Lumpur","Singapore":"Singapore","Mongolia":"Ulaanbaatar",
"North Korea":"Pyongyang","Syria":"Damascus","Jordan":"Amman","Lebanon":"Beirut","Israel":"Jerusalem","Qatar":"Doha","Oman":"Muscat","Yemen":"Sanaa"}
langs={"Mexico":"Spanish","Germany":"German","Japan":"Japanese","Egypt":"Arabic","Iran":"Persian","Russia":"Russian","Italy":"Italian","Greece":"Greek","Turkey":"Turkish","Sweden":"Swedish","Poland":"Polish","Kenya":"Swahili","Israel":"Hebrew","Thailand":"Thai","Vietnam":"Vietnamese"}
curr={"the United Kingdom":"pound","the United States":"dollar","India":"rupee","China":"yuan","Russia":"ruble","Switzerland":"franc","Mexico":"peso","South Korea":"won","Brazil":"real","Poland":"zloty","Turkey":"lira","Sweden":"krona"}
elements={"oxygen":"O","hydrogen":"H","carbon":"C","nitrogen":"N","silver":"Ag","copper":"Cu","potassium":"K","calcium":"Ca","lead":"Pb","mercury":"Hg","tin":"Sn","zinc":"Zn","helium":"He","neon":"Ne","uranium":"U","chlorine":"Cl","sulfur":"S","magnesium":"Mg"}
books={"Romeo and Juliet":"Shakespeare","Don Quixote":"Cervantes","The Divine Comedy":"Dante","Crime and Punishment":"Dostoevsky","Moby-Dick":"Melville","Ulysses":"Joyce",
"The Great Gatsby":"Fitzgerald","One Hundred Years of Solitude":"Garcia Marquez","Les Miserables":"Hugo","The Iliad":"Homer","Faust":"Goethe","Frankenstein":"Shelley",
"Oliver Twist":"Dickens","The Old Man and the Sea":"Hemingway","Animal Farm":"Orwell","The Hobbit":"Tolkien","Harry Potter":"Rowling","Brave New World":"Huxley","The Trial":"Kafka","Anna Karenina":"Tolstoy"}
people=[("Who painted the Sistine Chapel ceiling?","Michelangelo"),("Who painted Guernica?","Picasso"),("Who invented the light bulb?","Edison"),("Who discovered radium?","Curie"),
("Who formulated the laws of motion?","Newton"),("Who composed The Magic Flute?","Mozart"),("Who was the first man on the Moon?","Armstrong"),("Who wrote the Communist Manifesto with Engels?","Marx"),
("Who was the first emperor of Rome?","Augustus"),("Who led India's independence movement with nonviolence?","Gandhi"),("Who developed the polio vaccine?","Salk"),("Who invented the World Wide Web?","Berners-Lee"),
("Who was the British prime minister during most of World War II?","Churchill"),("Who sculpted David?","Michelangelo"),("Who wrote the Origin of Species?","Darwin"),("Who discovered America in 1492?","Columbus"),
("Who painted The Last Supper?","Leonardo da Vinci"),("Who invented the printing press?","Gutenberg"),("Who proposed the heliocentric model?","Copernicus"),("Who composed The Four Seasons?","Vivaldi")]
companies=[("Who is the CEO of Apple?","Tim Cook"),("Who is the CEO of Microsoft?","Satya Nadella"),("Who is the CEO of Alphabet?","Sundar Pichai"),("Who is the CEO of Amazon?","Andy Jassy"),
("Who is the CEO of Meta?","Mark Zuckerberg"),("Who founded Tesla Motors' rival SpaceX?","Elon Musk"),("Who founded Oracle?","Larry Ellison"),("Who founded Dell?","Michael Dell"),
("Who is the head of Berkshire Hathaway?","Warren Buffett"),("Who founded Alibaba?","Jack Ma"),("Who founded Nintendo?","Fusajiro Yamauchi"),("Who founded Ford Motor Company?","Henry Ford"),
("In which city is Google's headquarters?","Mountain View"),("In which city is Amazon's headquarters?","Seattle"),("In which city is BMW's headquarters?","Munich"),("In which city is Sony's headquarters?","Tokyo"),
("In which city is Siemens' headquarters?","Munich"),("In which city is Nokia's headquarters?","Espoo"),("In which city is Philips' headquarters?","Amsterdam"),("In which city is Hyundai's headquarters?","Seoul"),
("In which city is Coca-Cola's headquarters?","Atlanta"),("In which city is Starbucks' headquarters?","Seattle"),("In which city is Nike's headquarters?","Beaverton"),("In which city is Airbus headquartered?","Leiden"),
("Where is Intel headquartered?","Santa Clara"),("Where is Netflix headquartered?","Los Gatos"),("In what year was IBM founded?","1911"),("In what year was Facebook founded?","2004"),
("In what year was Tesla founded?","2003"),("In what year was Netflix founded?","1997"),("In what year was Nintendo founded?","1889"),("When was Coca-Cola founded?","1892"),
("In what year was Toyota founded?","1937"),("In what year was Samsung founded?","1938"),("What is the stock ticker of Alphabet?","GOOGL"),("What is the stock ticker of Nvidia?","NVDA"),
("What is the stock ticker of Meta?","META"),("What is the stock ticker of Netflix?","NFLX"),("What is the stock ticker of Intel?","INTC"),("What is the ticker symbol of Coca-Cola?","KO"),
("What is the stock ticker of IBM?","IBM"),("What is the ticker symbol of Ford?","F"),("What animal is the mascot of Firefox?","fox"),("What animal is in the logo of Twitter?","bird"),
("What animal is the logo of Lamborghini?","bull"),("What animal is the symbol of Peugeot?","lion"),("What animal is the mascot of Python?","snake"),("What is the name of Google's AI chatbot?","Gemini"),
("What is the name of OpenAI's chatbot?","ChatGPT"),("What is the name of Microsoft's AI assistant?","Copilot"),("What is the name of Samsung's voice assistant?","Bixby"),("What is the name of Apple's desktop operating system?","macOS"),
("What is the name of Amazon's e-reader?","Kindle"),("What is the name of Microsoft's game console?","Xbox"),("What is the name of Google's web browser?","Chrome"),("What is the name of Apple's smartwatch?","Apple Watch"),
("What is the codename of Android 10?","Q"),("Which company makes the Galaxy phones?","Samsung"),("Which company makes Windows?","Microsoft"),("Which company makes the Model 3?","Tesla"),
("Where is Samsung's main semiconductor factory?","Giheung"),("Which city hosts the headquarters of the United Nations?","New York"),("Where is the headquarters of the European Union?","Brussels"),
("Which city hosts the headquarters of FIFA?","Zurich"),("Which city hosts the International Olympic Committee?","Lausanne"),("Who leads the Catholic Church?","the Pope")]
misc=[("What is the boiling point of water in Celsius?","100"),("How many sides does a hexagon have?","6"),("How many planets are in the solar system?","8"),("What is the largest country by area?","Russia"),
("What is the smallest country in the world?","Vatican City"),("What is the longest river in South America?","Amazon"),("What is the highest waterfall in the world?","Angel Falls"),("What is the largest island in the world?","Greenland"),
("Which ocean lies between Africa and Australia?","Indian"),("What is the deepest ocean trench?","Mariana"),("What is the main gas in Earth's atmosphere?","nitrogen"),("What organ pumps blood?","heart"),
("How many bones are in the adult human body?","206"),("What is the largest organ of the human body?","skin"),("What is the speed of light in km per second?","300000"),("What is H2O?","water"),
("What planet is known as the Red Planet?","Mars"),("Which planet has the most prominent rings?","Saturn"),("What is the closest star to Earth?","Sun"),("What is the largest bird?","ostrich"),
("What is the tallest animal?","giraffe"),("What do bees make?","honey"),("What is a baby kangaroo called?","joey"),("Which animal is known as the ship of the desert?","camel"),
("What is the largest cat?","tiger"),("What animal is the symbol of the WWF?","panda"),("In what year did World War I begin?","1914"),("In what year did the French Revolution begin?","1789"),
("In what year did the Titanic sink?","1912"),("In what year did Columbus reach the Americas?","1492"),("In what year was the Declaration of Independence signed?","1776"),("In what year did the Soviet Union collapse?","1991"),
("In what year did the 2008 financial crisis start?","2008"),("What year did the first iPhone come out?","2007"),("In what city were the 2012 Summer Olympics held?","London"),("Which city is known as the Big Apple?","New York"),
("In which city is the Eiffel Tower?","Paris"),("In which city is the Statue of Liberty?","New York"),("In which city is the Taj Mahal?","Agra"),("In which city is the Kremlin?","Moscow"),
("In which city is the Sagrada Familia?","Barcelona"),("In which city is Big Ben?","London"),("In which city is the Brandenburg Gate?","Berlin"),("In which city is the Acropolis?","Athens"),
("In which country is Kyoto?","Japan"),("In which country is Marrakesh?","Morocco"),("In which country is Wellington?","New Zealand"),("In which country is Kaohsiung located?","Taiwan"),
("What is 7 times 8?","56"),("What is 9 times 9?","81"),("What is 14 plus 29?","43"),("What is 50 minus 17?","33"),("What is 3 cubed?","27"),("What is 2 to the power of 10?","1024"),
("What is half of 90?","45"),("What is 18 divided by 3?","6"),("What is 123 plus 456?","579"),("What is the square root of 144?","12"),("How many minutes are in an hour?","60"),("How many hours are in a day?","24")]
GQ=[(f"What is the capital of {k}?",v) for k,v in capitals.items()]+[(f"What language is spoken in {k}?",v) for k,v in langs.items()]+[(f"What currency is used in {k}?",v) for k,v in curr.items()]
GQ+=[(f"What is the chemical symbol for {k}?",v) for k,v in elements.items()]+[(f"Who wrote {k}?",v) for k,v in books.items()]+people+companies+misc
print('general', len(GQ))
prose=["The river wound slowly through the valley, past old stone bridges and fields of barley that turned gold in late summer.",
"Photosynthesis converts light energy into chemical energy, allowing plants to produce glucose from carbon dioxide and water.",
"In 1969, the Apollo 11 mission landed on the Moon, and Neil Armstrong became the first person to walk on its surface.",
"To install the package, open a terminal, navigate to the project folder, and run the setup script with administrator rights.",
"The quarterly report showed revenue growth of eight percent, driven mainly by strong sales in the European market.",
"Mix the flour, sugar and butter until the dough is smooth, then let it rest in the refrigerator for at least an hour.",
"The committee will meet on Tuesday to discuss the budget proposal and vote on the new hiring policy for next year.",
"Mount Everest, located on the border between Nepal and China, is the highest mountain above sea level on Earth.",
"Our company was founded by two engineers who wanted to build reliable software for small businesses around the world.",
"The head of marketing presented a new campaign, while the finance team reviewed the costs and the legal department checked the contracts.",
"Each employee receives a badge, access to the internal wiki, and an invitation to the annual company retreat in the spring.",
"The museum's collection includes paintings by Rembrandt, Vermeer and Van Gogh, as well as ancient Egyptian artifacts.",
"Researchers at the university discovered a new species of frog in the rainforest, bright blue with yellow spots.",
"The train to Vienna departs at 9:15 from platform four; passengers should arrive at least ten minutes early.",
"Python is a popular programming language known for its readable syntax and large ecosystem of libraries.",
"The startup opened a new office in Berlin and plans to hire fifty engineers and designers over the next two years.",
"During the storm, the power went out across the city and emergency crews worked through the night to restore it.",
"The novel tells the story of a young woman who leaves her village to study medicine in the capital.",
"Support tickets are assigned a number and routed to the appropriate team, which responds within one business day.",
"The stock market closed higher on Friday as technology shares rallied after strong earnings from several large firms.",
"Shakespeare wrote many famous plays, including Hamlet, Macbeth, Othello and King Lear, during the late sixteenth century.",
"The factory produces electric motors and batteries, shipping them to car manufacturers across Asia and Europe.",
"Our data center uses renewable energy and advanced cooling systems to reduce its environmental impact.",
"The mascot of the football club is a bear, which appears on the team's jersey and at every home game.",
"Tokyo is the capital of Japan and one of the largest metropolitan areas in the world, with over thirty million residents.",
"The conference room on the third floor has a projector, a whiteboard and seating for twenty people.",
"The internal codename for the project was kept secret until the official launch event in the autumn.",
"Water boils at one hundred degrees Celsius at sea level, but at lower temperatures at high altitude.",
"The newsletter is sent to all employees every Monday and includes updates from each department.",
"Guests can connect to the Wi-Fi network using the password printed on the card at the reception desk."]
TB=[];MB=[]
for q,a in GQ: TB.append(prompt(q)+a); MB.append(('G',q,a))
random.seed(3)
for f in facts:
    combos=random.sample([(c,w) for c in NEG_C for w in range(12)],15)
    for c,w in combos: TB.append(prompt(f['tmpl'][w].replace('{C}',c))); MB.append(('O',f['id'],c,w))
for s in prose: TB.append(s); MB.append(('P',s))
lb=[len(tok.encode(t,add_special_tokens=False).ids) for t in TB]
print(len(TB), max(lb), sum(lb), len(TB)*max(lb)*48/1e6,'GB')
```