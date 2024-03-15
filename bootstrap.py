#NOTE: WORDNET KEY IS IN LEMMA SENSE KEYS
import os
import xml.etree.ElementTree as ET
from statistics import mean

import nltk
import numpy as np
#import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.cm as cm

from nltk.corpus import wordnet as wn
from sklearn.metrics import accuracy_score

from loader import load_instances, load_key
from sklearn.svm import SVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder

from datasets import load_dataset
from sklearn.feature_extraction.text import  CountVectorizer
from collections import Counter, defaultdict


#my hypothesis is that the bug is from using a diffrent wordnet. the dev_keys and test_keys are from profs (the right) file, not wn.

#TODO: get the feature extraction working for the seed sets and for the x_train
sense_sentences_dict_game = {
    "game.n.01": [
        "The chess game challenged their strategic thinking.",
        "Every weekend, they organized a card game.",
        "The final game of the tournament was intensely competitive.",
        "Children gathered in the park for a game of tag.",
        "The game of trivia tested their general knowledge.",
        "They enjoyed a friendly game of soccer in the backyard.",
        "The game of monopoly lasted for hours.",
        "A game of skill, like chess, requires patience.",
        "They engaged in a game of charades at the party.",
        "The game ended with a surprising victory."
    ],

    "game.n.02": [
        "Tonight's football game attracted a large audience.",
        "The basketball game went into overtime.",
        "During the baseball game, he hit a home run.",
        "The first game of the season set the tone for the team.",
        "She scored the winning goal in the hockey game.",
        "The rugby game was postponed due to weather.",
        "Fans cheered throughout the exciting volleyball game.",
        "The tennis game was paused due to rain.",
        "In the cricket game, he achieved his highest score.",
        "The boxing game was intense from start to finish."
    ],

    "game.n.03": [
        "Their favorite game was solving crossword puzzles.",
        "Family gatherings often included a game of charades.",
        "The kids spent the afternoon playing video games.",
        "They developed a new board game for language learners.",
        "Playing a guessing game, they laughed for hours.",
        "At the camp, children played a variety of outdoor games.",
        "The party included a game of musical chairs.",
        "On long car trips, they played word games.",
        "He invented a game to make learning math fun.",
        "The game involved matching pairs of cards."
    ],

    "game.n.04": [
        "They went into the woods to hunt game.",
        "Deer are a common type of game in this region.",
        "The forest was rich with game, like rabbits and pheasants.",
        "Hunting game was a traditional practice in the village.",
        "He learned to track game from his grandfather.",
        "The game birds were a challenging hunt.",
        "In ancient times, people relied on game for sustenance.",
        "The game reserves protect wildlife from poaching.",
        "During the safari, they observed various game animals.",
        "They obtained a license for game hunting."
    ],

    "game.n.05": [
        "He won the first game of the set with an ace.",
        "Serving for the game, she felt the pressure.",
        "Each game in the match was intensely contested.",
        "They practiced playing games to improve their serve.",
        "The game went to deuce several times.",
        "In tennis, scoring a game requires strategy.",
        "He lost the game on a double fault.",
        "Winning the game, she gained momentum.",
        "The crowd applauded after a well-played game.",
        "They analyzed each game to improve their tactics."
    ],

    "game.n.06": [
        "The game was tied at 30-all.",
        "He was one point away from winning the game.",
        "They kept a close eye on the game score.",
        "In the final game, the tension was palpable.",
        "She checked the scoreboard to see the game status.",
        "Winning the next game would secure the championship.",
        "The game was in his favor, but he lost focus.",
        "The game score reflected their skill levels.",
        "She celebrated each game won against her rival.",
        "The game point was a crucial moment in the match."
    ],

    "game.n.07": [
        "They prepared a stew with game meat.",
        "Game like venison was a delicacy in the region.",
        "The chef specialized in dishes made from game.",
        "Game provided a rich, distinct flavor to the meal.",
        "They learned to cook game on an open fire.",
        "The market sold various types of game.",
        "He preferred the taste of game over domestic meat.",
        "Game was a staple in their traditional cuisine.",
        "They cured the game for preservation.",
        "The game pie was a popular dish at the feast."
    ],

    "game.n.08": [
        "They unraveled his game of corporate espionage.",
        "Her game of deceit was finally exposed.",
        "He was known for playing a dangerous game in business.",
        "The game involved manipulating the stock market.",
        "Their game was foiled by the investigators.",
        "He found himself unwittingly caught in their game.",
        "The political game led to unexpected consequences.",
        "They plotted a game to embezzle funds.",
        "His game of fraud lasted for years before being discovered.",
        "She realized too late that she was part of their game."
    ],

    "game.n.09": [
        "They set up the game for a night of fun.",
        "He packed all the game pieces carefully.",
        "The game included cards, dice, and a board.",
        "She lost a crucial piece of the game.",
        "The new game came with detailed instructions.",
        "They designed a game that required strategy and luck.",
        "The game was popular among children and adults alike.",
        "He collected vintage game sets as a hobby.",
        "The game required both skill and teamwork.",
        "They spent hours engaged in the tabletop game."
    ],

    "game.n.10": [
        "He's been in the real estate game for years.",
        "She knew the publishing game inside out.",
        "They were new to the restaurant game.",
        "He made a name for himself in the tech game."
        "She wanted to break into the fashion game.",
        "They learned the hard way how the political game is played.",
        "In the finance game, risks can lead to high rewards.",
        "He was a veteran in the journalism game.",
        "Breaking into the movie game requires persistence.",
        "She quickly adapted to the marketing game.",
    ],

    "game.n.11": [
        "He was tired of their game of avoiding responsibility.",
        "Her game of playing hard to get was obvious.",
        "They were caught up in a game of office politics.",
        "His game of evading questions frustrated the interviewer.",
        "The children's game of hide-and-seek brought joy.",
        "Their game of one-upmanship became tiresome.",
        "She didn't appreciate his game of playing the fool.",
        "The game of cat and mouse between the detective and thief was intense.",
        "He realized the game of false flattery being played.",
        "Their game of misleading each other led nowhere.",
    ],


}


sense_sentences_dict_year = {
    "year.n.01": [
        "The calendar year starts on January 1st.",
        "He moved to the city two years ago.",
        "The year 2020 was marked by global events.",
        "She celebrates her birthday every year with a big party.",
        "The company's fiscal year differs from the calendar year.",
        "Every leap year has an extra day in February.",
        "They traveled around the world for a whole year.",
        "The yearly cycle of seasons is fascinating.",
        "She takes a year-end vacation to relax.",
        "The year of her graduation was a memorable one."
    ],

    "year.n.02": [
        "The academic year begins in September.",
        "The financial year is crucial for budget planning.",
        "Many sports have their off-season during the year.",
        "The agricultural year is busy during planting and harvest times.",
        "The fiscal year determines the accounting period.",
        "Schools often celebrate the end of the school year with events.",
        "Their organization plans activities on a year-round basis.",
        "The legislative year was marked by significant policy changes.",
        "During the tourist year, the town's population doubles.",
        "The year in the wine industry is marked by the grape harvest."
    ],

    "year.n.03": [
        "A year on Earth is slightly longer than 365 days.",
        "Jupiter's year is much longer than Earth's.",
        "Scientists calculate a year on Venus differently from Earth's year.",
        "One year on Saturn is equivalent to about 29.5 Earth years.",
        "In astronomy, a year is a crucial time unit.",
        "Mars completes a year in approximately 687 Earth days.",
        "Studying a planet's year helps understand its seasons.",
        "The concept of a year varies from planet to planet.",
        "A year on Mercury lasts just 88 Earth days.",
        "The length of a year is key in defining a planet's orbit."
    ],

    "year.n.04": [
        "The year 2020 faced unprecedented challenges.",
        "She was part of the graduating year of 2018.",
        "The entire year participated in the graduation ceremony.",
        "He was voted the most likely to succeed in his year.",
        "Their year organized a reunion ten years later.",
        "The yearbook featured photos of the entire year.",
        "She was the top student in her year.",
        "The year gathered for their final school assembly.",
        "Members of the year stayed in touch through social media.",
        "The year had a memorable senior trip."
    ],
}

sense_sentences_dict_country = {
    "country.n.01": [
        "The country's new policy aims to improve public health.",
        "Elections are a significant event in any democratic country.",
        "The country faced economic challenges after the global crisis.",
        "She aspired to serve her country in the government.",
        "The country's constitution was amended to promote equality.",
        "Tourists were fascinated by the history of the country.",
        "The country celebrated its independence day with great enthusiasm.",
        "He was a true patriot who loved his country deeply.",
        "The country's growth rate has been steady over the past decade.",
        "Diplomats from various countries met to discuss global issues."
    ],

    "country.n.02": [
        "The river forms the natural border of the country.",
        "They traveled across the country on a road trip.",
        "The country's landscape is known for its scenic beauty.",
        "Natural resources are abundant in this part of the country.",
        "The country boasts a diverse range of flora and fauna.",
        "Historical landmarks are scattered throughout the country.",
        "They decided to explore the remote areas of the country.",
        "The country is famous for its picturesque countryside.",
        "Severe weather conditions affected large parts of the country.",
        "The country's geography influences its climate patterns."
    ],

    "country.n.03": [
        "The country's population is known for its cultural diversity.",
        "Every citizen in the country has the right to vote.",
        "The country united to celebrate the national festival.",
        "A survey was conducted to understand the mood of the country.",
        "The country's youth are driving the new technological revolution.",
        "Traditional customs are still practiced by the people of the country.",
        "The country was in mourning after the loss of a beloved leader.",
        "The entire country participated in the environmental conservation efforts.",
        "Educational reforms were widely welcomed by the country.",
        "The country's workforce is one of the most skilled in the world."
    ],

    "country.n.04": [
        "They bought a house in the country for a peaceful life.",
        "Farming is a common occupation in the country.",
        "The country offers a serene escape from urban hustle.",
        "They spent their weekends hiking in the country.",
        "Country life is characterized by a close-knit community.",
        "The beauty of the country is unspoiled by industrialization.",
        "Country roads are less crowded and more scenic.",
        "Many artists find inspiration in the tranquility of the country.",
        "Local festivals are a highlight of life in the country.",
        "The country has its own unique charm and appeal."
    ],

    "country.n.05": [
        "The country is known for its unique cultural heritage.",
        "They specialize in cuisines from a specific country.",
        "This country is the hub for technological advancements.",
        "The country has a rich history dating back centuries.",
        "They conducted geological studies in the desert country.",
        "The country is facing environmental challenges due to climate change.",
        "Tourists flock to this country for its famous landmarks.",
        "He wrote extensively about the traditions of this country.",
        "The wine country is famous for its exquisite vineyards.",
        "The country is undergoing rapid urban development."
    ],


}

sense_sentences_dict_action = {

    "action.n.01": [
        "The committee took immediate action to address the issue.",
        "Her decisive action prevented a disaster.",
        "He regretted his hasty action in the heat of the moment.",
        "The organization demands action, not just promises.",
        "Their actions spoke louder than their words.",
        "After much deliberation, she finally took action.",
        "The action of calling for help was crucial.",
        "His brave actions saved the day.",
        "The actions of a few can impact many.",
        "She valued action over lengthy discussions."
    ],

    "action.n.02": [
        "Her day was filled with constant action and movement.",
        "The action in the market today was unprecedented.",
        "He was admired for his relentless action and energy.",
        "The city center was always a hub of action.",
        "The organization was in action from dawn till dusk.",
        "The festival was a scene of lively action.",
        "She thrived in environments full of action.",
        "Their business was back in action after the renovation.",
        "The action of the crowd indicated excitement.",
        "He was never one to avoid action."
    ],

    "action.n.03": [
        "The troops prepared for action at dawn.",
        "He received a medal for his bravery in action.",
        "The battle was her first experience of action.",
        "The platoon saw heavy action during the campaign.",
        "Reports from the front lines indicated intense action.",
        "Veterans recounted stories of action during the war.",
        "The regiment was renowned for its action in combat.",
        "The action on the eastern front was escalating.",
        "Soldiers often speak of the brotherhood formed in action.",
        "The history book detailed the action of the famous battle."
    ],

    "action.n.04": [
        "The action of the tides is consistent and predictable.",
        "Erosion is the result of the action of wind and water.",
        "The volcanic action formed new islands.",
        "Photosynthesis is an essential action in plants.",
        "The action of natural selection shapes species.",
        "Stalactites are formed by the action of water over centuries.",
        "The action of the sun on the skin can be harmful.",
        "Gravity's action is a fundamental force in the universe.",
        "The action of frost can cause rocks to split.",
        "The action of bacteria is crucial in decomposition."
    ],

    "action.n.05": [
        "The movie's action kept the audience on the edge of their seats.",
        "The book's action unfolds in a post-apocalyptic world.",
        "The play's action revolves around a family drama.",
        "In the novel, the action switches between past and present.",
        "The action in the second act was riveting.",
        "She enjoys stories with fast-paced action.",
        "The action of the story is set in a small village.",
        "The narrative's action involves a complex political intrigue.",
        "Throughout the action, the protagonist evolves significantly.",
        "The action culminates in an unexpected twist."
    ],

    "action.n.06": [
        "His action in the face of danger was commendable.",
        "She's known for her action-oriented approach.",
        "The leader's action inspired the whole team.",
        "He's a man of action in all his endeavors.",
        "Her action in restructuring the company was effective.",
        "The project needs someone with a strong action bias.",
        "Their action-driven strategy yielded quick results.",
        "He has a reputation for bold action and decisiveness.",
        "The committee required more action and less talk.",
        "Her action in resolving conflicts is always admired."
    ],

    "action.n.07": [
        "The action of the clock was intricate and precise.",
        "He adjusted the action of the machine for better performance.",
        "The piano's action needed repair to improve its sound.",
        "In the engine, the action of the pistons is synchronized.",
        "The action of the lever controls the entire apparatus.",
        "Technicians checked the action of the automated system.",
        "The gun's action was jammed and required maintenance.",
        "She demonstrated the action of the new device to the team.",
        "The action of the gears was smooth and efficient.",
        "The robot's action was programmed for precision tasks."
    ],

    "action.n.08": [
        "The legal action against the company was initiated last month.",
        "He filed an action for breach of contract.",
        "The court dismissed the action due to lack of evidence.",
        "The action was settled out of court.",
        "The plaintiff's action sought damages for negligence.",
        "The class-action lawsuit involved hundreds of plaintiffs.",
        "Her lawyer advised her on the best course of action.",
        "The action against the manufacturer alleged faulty products.",
        "The action at law was a landmark case.",
        "The judge ruled that the action had merit and would proceed."
    ],

    "action.n.09": [
        "The government's action on climate change was applauded.",
        "The United Nations took action to address the crisis.",
        "The action taken by the council was unprecedented.",
        "Emergency action was required to deal with the disaster.",
        "The senate's action on the bill was delayed.",
        "The committee proposed action to reform the system.",
        "Global action is needed to combat the pandemic.",
        "The president announced decisive action on the issue.",
        "The regulatory body's action impacted many industries.",
        "Joint action by the member states was agreed upon."
    ],

    "action.n.10": [
        "The action in the tech industry is shifting towards AI.",
        "Wall Street is where the financial action happens.",
        "Silicon Valley is the hub of action for startups.",
        "The action in contemporary art is moving towards digital mediums.",
        "The real estate market is seeing a lot of action lately.",
        "The action in scientific research is advancing rapidly.",
        "Hollywood is where the action is for aspiring actors.",
        "The action in renewable energy is gaining momentum.",
        "The literary world's action is centered around new authors.",
        "The city's downtown is where the cultural action is."
    ]

}

sense_sentences_dict_claim = {

    "claim.n.01": [
        "The insurance company disputed his claim for compensation.",
        "She filed a claim for lost wages due to the accident.",
        "His claim on the estate was backed by a legal will.",
        "The heirs made conflicting claims to the family fortune.",
        "Her claim for the refund was processed quickly.",
        "They lodged a claim against the company for negligence.",
        "He relinquished his claim to the property after the settlement.",
        "The claim for damages was settled out of court.",
        "Several claimants came forward to assert their claims.",
        "The lawsuit revolves around a disputed land claim."
    ],

    "claim.n.02": [
        "The scientist's claim about the new discovery sparked debate.",
        "Her claim of being the first in her family to graduate was true.",
        "The advertisement's claim about the product's effectiveness was misleading.",
        "He refuted the claim that he was involved in the scandal.",
        "Their claim of innocence was supported by new evidence.",
        "The politician's claim was met with skepticism.",
        "She stood by her claim despite the criticism.",
        "The author's claim was later proven to be false.",
        "There was no evidence to back his claim.",
        "The claim that the diet was a cure-all was exaggerated."
    ],

    "claim.n.03": [
        "Workers voiced their claim for better working conditions.",
        "The union presented its claim for higher wages.",
        "Their claim for equal rights was heard in the highest court.",
        "He pressed his claim for recognition of the invention.",
        "The protestors made a strong claim for environmental justice.",
        "Her claim for a promotion was based on years of hard work.",
        "They made a moral claim for the return of the artifacts.",
        "The claim for improved public services grew louder.",
        "She asserted her claim for fair treatment.",
        "Their claim for reparations was long overdue."
    ],

    "claim.n.04": [
        "He had a longstanding claim on her affections.",
        "Her claim to the leading role was undisputed.",
        "His claim on the team's loyalty was evident.",
        "She felt a personal claim to the family heirlooms.",
        "His claim to the parking spot was based on seniority.",
        "She exercised her claim over the decision-making process.",
        "His claim on the community's respect was well-deserved.",
        "Her claim to the secret recipe was part of family lore.",
        "The coach's claim on the players' admiration was clear.",
        "He had a claim to the nickname due to his quirky habits."
    ],

    "claim.n.05": [
        "The company's legal claim to the patent was upheld.",
        "He had a solid claim to the inheritance under the law.",
        "Her claim to the throne was recognized by the council.",
        "The museum had a rightful claim to the historic artifacts.",
        "They had a legitimate claim to the land based on the treaty.",
        "Her claim to the title was validated by the committee.",
        "The nation's claim to the territory was internationally recognized.",
        "He established his claim to the record with official documentation.",
        "Their claim to the trademark was undisputed.",
        "She defended her claim to the intellectual property."
    ],

    "claim.v.01": [
        "He claimed to have witnessed the incident firsthand.",
        "She confidently claimed her idea would change the industry.",
        "The scientist claimed that his theory was revolutionary.",
        "The advertisement claimed the product could solve all problems.",
        "They claimed to have reached the summit despite the odds.",
        "He claimed that he had been unfairly treated.",
        "She claimed mastery over three languages.",
        "The author claimed that the story was based on real events.",
        "The politician claimed to represent the people's interests.",
        "They claimed to have found the ancient artifact."
    ],

    "claim.v.02": [
        "He claimed his rightful inheritance after years of dispute.",
        "The artist claimed the painting as her own creation.",
        "They claimed the land based on their ancestral rights.",
        "She claimed the title of champion after her victory.",
        "The group claimed responsibility for the successful project.",
        "He claimed ownership of the idea during the meeting.",
        "The family claimed the estate as per the will.",
        "She claimed the throne as the rightful heir.",
        "The company claimed the patent for the new invention.",
        "He claimed the prize he had won in the contest."
    ],
    "claim.v.03": [
        "The creditor claimed the unpaid debts from the debtor.",
        "She claimed reimbursement for her travel expenses.",
        "The tenants claimed the security deposit back from the landlord.",
        "He claimed compensation for the damage caused.",
        "The company claimed its dues from the client.",
        "She claimed her legal fees after the court case.",
        "They claimed the refund for the faulty product.",
        "The heirs claimed their share of the inheritance.",
        "The organization claimed funding from the government grant.",
        "He claimed back the loan he had given his friend."
    ],
    "claim.v.05": [
        "The severe storm claimed several homes in the area.",
        "The epidemic claimed thousands of lives.",
        "The harsh winter claimed many crops.",
        "The financial crisis claimed numerous businesses.",
        "The relentless workload claimed her health.",
        "The fire claimed an entire block of the historic district.",
        "The war claimed many young soldiers.",
        "The treacherous mountain pass claimed several climbers.",
        "The economic downturn claimed numerous jobs.",
        "The famine claimed lives in the affected regions."
    ],


}


sense_sentences_dict_support = {
    "support.n.11": [
        "The startup received significant support from a local venture capital firm to launch its new app.",
        "Her research on climate change was made possible by the generous support of an environmental foundation.",
        "The community project was successful thanks to the support from various local businesses.",
        "The artist's new exhibition was a hit, largely due to the financial support of art enthusiasts.",
        "Without the necessary financial support, the ambitious infrastructure project could not get off the ground.",
        "The support from the government grant allowed the scientists to continue their groundbreaking research.",
        "The crowdfunding campaign aimed to gather enough support to publish the author's first novel.",
        "With the support of an angel investor, the small company quickly expanded its operations.",
        "The film festival was able to feature more independent films this year, thanks to increased financial backing.",
        "The charity organization relied heavily on the financial support of its donors to carry out its missions."
    ],
    "support.n.02": [
        "The new policy has gained a lot of support among young voters.",
        "She expressed her strong support for educational reform during the meeting.",
        "Despite initial resistance, the healthcare initiative eventually received widespread support.",
        "The support of the community was crucial in making the local recycling program a success.",
        "His unwavering support for environmental causes inspired many others to take action.",
        "The support from the union played a key role in the candidate’s election victory.",
        "The president's latest decision has eroded the support of several key allies.",
        "After the debate, the politician saw an increase in support from undecided voters.",
        "The campaign relied on the grassroots support of thousands of volunteers.",
        "The new law passed with the support of both parties, marking a rare moment of bipartisan cooperation."
    ],
}


sense_sentences_dict_law = {
    "law.n.01": [
        "Respect for the law is a fundamental principle of a civilized society.",
        "The field of jurisprudence continuously evolves to meet the changing needs of society.",
        "Breaking the law can lead to serious consequences, including imprisonment.",
        "The lawyer's deep understanding of the law helped her win many cases.",
        "The law serves to maintain order and protect the rights of citizens.",
        "Many activists work to change outdated laws and promote justice.",
        "The constitution is the foundational law of the land.",
        "Law enforcement agencies are tasked with upholding the law and keeping peace.",
        "International law governs relations between different countries.",
        "The supreme court has the final say in interpreting the law."
    ],
    "law.n.02": [
        "The new law against texting while driving aims to reduce accidents.",
        "A law was passed to regulate the sale of tobacco to minors.",
        "The company had to comply with international trade laws.",
        "Local zoning laws dictate how land can be used in different areas.",
        "Environmental laws are crucial for protecting our natural resources.",
        "The law stipulates that all employees must receive equal pay for equal work.",
        "Tax laws can be complex and vary significantly from country to country.",
        "The new privacy law aims to protect individuals' personal data online.",
        "Labor laws ensure that workers' rights are protected in the workplace.",
        "The law prohibits the use of certain substances in food products."
    ]
}


sense_sentences_dict_rule = {
    "rule.n.03": [
        "The company's dress code is a rule that all employees are expected to follow.",
        "Each game has its own set of rules that players must adhere to.",
        "The rule to turn off lights when leaving a room is a simple way to save energy.",
        "Parents often establish rules to guide their children's behavior.",
        "The library has a strict rule against talking loudly.",
        "In the laboratory, the rule is to always wear safety goggles.",
        "The teacher’s rule was to submit all assignments by the end of the day.",
        "Their family had a rule of no screen time during dinner.",
        "The golden rule of customer service is to always be polite and helpful.",
        "The rule of not running in the corridors is strictly enforced in the school."
    ],
    "rule.n.01": [
        "It's a general rule to wash your hands before meals.",
        "The rule of keeping left is followed in most countries while driving.",
        "In their household, it was a regulation to have dinner together.",
        "School regulations require students to wear uniforms.",
        "As a rule, he goes for a jog every morning.",
        "The company's regulations forbid the use of personal devices at work.",
        "Traffic rules are important for the safety of both drivers and pedestrians.",
        "The camp had a regulation of lights out by 10 pm.",
        "It's an unspoken rule in their club to help each other in times of need.",
        "The rule in the laboratory is to record every observation meticulously."
    ]
}


sense_sentences_dict_end = {
    "end.n.03": [
        "The end of the championship game was thrilling, with a last-minute goal.",
        "Everyone was on the edge of their seats during the end of the suspense movie.",
        "The last chapter of the book provided a satisfying conclusion to the story.",
        "She arrived just in time to catch the end of her favorite TV show.",
        "The end of the play was marked by a standing ovation from the audience.",
        "In the end, the hero triumphed over the villain in an epic battle.",
        "The concert ended on a high note, leaving the audience wanting more.",
        "He always reads the end of the novel first out of curiosity.",
        "The final moments of the lecture were particularly insightful.",
        "The end of the marathon was a test of endurance for the runners."
    ],
    "end.n.02": [
        "The end of the school year is always a busy time.",
        "She was relieved at the ending of her long work shift.",
        "The warranty period's end means we need to be careful with our equipment.",
        "They celebrated the end of 2020 with a small family gathering.",
        "The project's deadline was fast approaching its end.",
        "The end of the lease marked the time to search for a new apartment.",
        "At the end of the meeting, everyone agreed on the next steps.",
        "The film's ending left the audience in anticipation of a sequel.",
        "The end of the trial brought relief to everyone involved.",
        "He was looking forward to the end of his military service."
    ]
}


def find_top_words_with_senses(test_key, number):
    word_freq = Counter()
    word_senses = defaultdict(set)

    # Iterate through the test_key data
    for id, sense_data in test_key.items():
        word, sense_key = sense_data[0].split('%', 1)[0], sense_data[0]
        word_freq[word] += 1
        word_senses[word].add(sense_key)

    # Filter out words with only one sense
    words_with_multiple_senses = {word: senses for word, senses in word_senses.items() if len(senses) > 1}

    # Sort words by the number of different senses (ascending) and frequency (descending)
    sorted_words = sorted(words_with_multiple_senses, key=lambda w: (len(word_senses[w]), -word_freq[w]))

    # Return the top four words along with their senses
    top_words_with_senses = {word: word_senses[word] for word in sorted_words[:number]}
    return top_words_with_senses

def setup_and_load_data( ):
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('punkt')
    nltk.download('stopwords')

    lemmatizer = WordNetLemmatizer()


    stop_words = set(stopwords.words('english'))

    data_file = 'multilingual-all-words.en.xml'
    key_file = 'wordnet.en.key'

    dev_instances, test_instances = load_instances(data_file)
    dev_key, test_key = load_key(key_file)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    print("The number of dev instances: ",  len(dev_instances))  # number of dev instances
    print("The number of test instances: ", len(test_instances))  # number of test instances

    return dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words


#based on words in the dev set with multiple senses in word net AND are also in the test set frequently
def get_most_frequent_lemmas(dev_instances, test_instances, top_n=5):
    dev_lemma_freq = Counter()
    test_lemma_freq = Counter()
    lemma_sense_count = {}

    # Count lemma frequencies in the development set
    for instance in dev_instances.values():
        lemma = instance.lemma
        dev_lemma_freq[lemma] += 1

    # Count lemma frequencies in the test set
    for instance in test_instances.values():
        lemma = instance.lemma
        test_lemma_freq[lemma] += 1

    # Record the number of distinct senses (synsets) in WordNet for lemmas in the development set
    for lemma in dev_lemma_freq:
        lemma_sense_count[lemma] = len(wn.synsets(lemma))

    # Select lemmas that are frequent in both the development and test sets and have multiple senses in WordNet
    selected_lemmas = [lemma for lemma, count in dev_lemma_freq.items()
                       if lemma_sense_count[lemma] > 1 and test_lemma_freq[lemma] > 0]

    # Sort the selected lemmas based on their frequency in the test set
    selected_lemmas.sort(key=lambda lemma: test_lemma_freq[lemma], reverse=True)

    return selected_lemmas[:top_n]

class TrainInstance:
    def __init__(self, lemma, context, index):
        self.lemma = lemma      # Lemma of the word whose sense is to be resolved
        self.context = context  # List of all the words in the sentential context
        self.index = index      # Index of lemma within the context


from nltk.stem import WordNetLemmatizer

from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def load_and_filter_training_data(wsd_term, train_data_folder, include_brown=True, include_gigaword=True, max_per_wsd_term=250, use_lemmatization=False, remove_stop_words=False):
    training_instances = []
    term_count = 0
    lemmatizer = WordNetLemmatizer() if use_lemmatization else None
    stop_words = set(stopwords.words('english')) if remove_stop_words else None

    # Function to process a sentence and add TrainInstances
    def process_sentence(words):
        nonlocal term_count
        for index, word in enumerate(words):
            processed_word = lemmatizer.lemmatize(word) if lemmatizer else word
            if remove_stop_words and processed_word in stop_words:
                continue
            if processed_word == wsd_term:
                training_instances.append(TrainInstance(processed_word, words, index))
                term_count += 1
                if term_count >= max_per_wsd_term:
                    return True
        return False

    if include_brown:
        for filename in os.listdir(train_data_folder):
            if filename.endswith(".xml"):
                full_path = os.path.join(train_data_folder, filename)
                tree = ET.parse(full_path)
                root = tree.getroot()

                for sentence in root.iter('sentence'):
                    sentence_words = []
                    for elem in sentence:
                        if elem.tag == 'wf' or elem.tag == 'instance':
                            word_text = elem.text.lower()
                            if not stop_words or word_text not in stop_words:
                                word_text = lemmatizer.lemmatize(word_text) if lemmatizer else word_text
                                sentence_words.append(word_text)
                    if process_sentence(sentence_words):
                        return training_instances

    if include_gigaword:
        gigaword_dataset = load_dataset("gigaword", split='train')
        for article in gigaword_dataset:
            sentence = article['document'].lower()
            words = [lemmatizer.lemmatize(word) if lemmatizer else word for word in sentence.split() if
                     not stop_words or word not in stop_words]
            if process_sentence(words):
                return training_instances

    return training_instances


def count_seed_words(training_data):
    seed_word_counts = Counter()
    for instance in training_data:
        word = instance.lemma  # Assuming instance.lemma is the word you're interested in
        seed_word_counts[word] += 1
    return seed_word_counts

def get_word_senses_and_examples(terms):
    term_senses = {}
    for term in terms:
        synsets = wn.synsets(term)
        term_senses[term] = {}
        for synset in synsets:
            # Fetch one example for the sense, if available
            examples = synset.examples()
            example = examples[0] if examples else "No example available"
            term_senses[term][synset.name()] = {'definition': synset.definition(), 'example': example}
    return term_senses


def create_seed_set(term, lemmatizer=None, stop_words=None):
    seed_set = []
    synsets = wn.synsets(term)

    for synset in synsets:
        examples = synset.examples()
        if not examples:
            continue

        example = examples[0]
        words = example.split()
        features = {'sentence': ' '.join(words)}  # Create a single string of the sentence

        seed_set.append((features, synset.name()))

    return seed_set








def evaluate_scenarios(highest_avg_scenario, lowest_avg_scenario, test_key, test_instances, lemmatizer, stop_words, wsd_terms, train_data_folder, max_samples_list=[500, 1000], confidence_thresholds=[0.5, 0.6, 0.7, 0.8, 0.9], max_iters=[10, 50, 100], default_confidence_threshold=0.3, default_max_iter=32, default_max_samples=250):
    # Initialize results dictionary
    results = {
        'Lemmatization + Stop Words': {},
        'Stop Words Only': {},
        'No Preprocessing': {},
        'Lemmatization Only': {}
    }

    # Function to process a scenario
    def process_scenario(max_samples, confidence_threshold, max_iter, lemmatizer_option, stop_words_option):
        accuracies = {}
        for term in wsd_terms:
            # Load and filter training data for each term
            training_data_term = load_and_filter_training_data(term, train_data_folder, max_per_wsd_term= max_samples)

            # Bootstrap the classifier
            classifier, vectorizer, label_encoder = bootstrap_algorithm_for_term(term, training_data_term, lemmatizer_option, stop_words_option, confidence_threshold, max_iter)

            # Filter test instances and preprocess test data
            filtered_test_instances = filter_test_instances_by_term(test_instances, term)
            X_test = preprocess_and_vectorize_test_data(filtered_test_instances, vectorizer, lemmatizer_option, stop_words_option)

            # Filter the test_key and evaluate classifier
            filtered_test_key = {id: test_key[id] for id in filtered_test_instances}
            accuracy = evaluate_classifier(classifier, label_encoder, X_test, filtered_test_key, test_instances)

            accuracies[term] = accuracy
        return accuracies

    # Get preprocessing options based on the scenario
    def get_preprocessing_options(scenario):
        if scenario == 'Lemmatization + Stop Words':
            return lemmatizer, stop_words
        elif scenario == 'Stop Words Only':
            return None, stop_words
        elif scenario == "No Preprocessing":
            return None, None
        elif scenario == "Lemmatization Only":
            return lemmatizer, None
        else:
            print("ERROR in get_preprocessing_options")
            return None, None

    # Evaluate scenarios
    for scenario in [highest_avg_scenario, lowest_avg_scenario]:
        lemmatizer_option, stop_words_option = get_preprocessing_options(scenario)
        scenario_results = {}

        # Vary max_samples
        for max_samples in max_samples_list:
            accuracies = process_scenario(max_samples, default_confidence_threshold, default_max_iter, lemmatizer_option, stop_words_option)
            for term, accuracy in accuracies.items():
                results[scenario].setdefault(term, []).append(f"max_samples={max_samples}, accuracy={accuracy}")

        # Vary confidence_thresholds
        for confidence_threshold in confidence_thresholds:
            accuracies = process_scenario(default_max_samples, confidence_threshold, default_max_iter, lemmatizer_option, stop_words_option)
            for term, accuracy in accuracies.items():
                results[scenario].setdefault(term, []).append(f"confidence_threshold={confidence_threshold}, accuracy={accuracy}")

        # Vary max_iters
        for max_iter in max_iters:
            accuracies = process_scenario(default_max_samples, default_confidence_threshold, max_iter, lemmatizer_option, stop_words_option)
            for term, accuracy in accuracies.items():
                results[scenario].setdefault(term, []).append(f"max_iter={max_iter}, accuracy={accuracy}")

    for scenario in [highest_avg_scenario, lowest_avg_scenario]:
        plot_scenario(results[scenario], "max_samples", max_samples_list, scenario)
        plot_scenario(results[scenario], "confidence_threshold", confidence_thresholds, scenario)
        plot_scenario(results[scenario], "max_iter", max_iters, scenario)


def plot_scenario(scenario_results, parameter, parameter_values, scenario_name):
    # Initialize the figure
    plt.figure(figsize=(12, 6))

    # Set the bar width based on the number of terms and parameter values
    num_terms = len(scenario_results)
    total_width = 0.8
    bar_width = total_width / num_terms

    viridis = cm.get_cmap('viridis', num_terms)


    # Set colors for each term
    colors = viridis(np.linspace(0, 1, num_terms))

    # Initialize data storage for accuracies and averages
    term_accuracies = {term: [] for term in scenario_results.keys()}
    avg_accuracies = []

    # Iterate over each parameter value and extract the corresponding accuracies
    for i, param_value in enumerate(parameter_values):
        mid_point = i  # This will be used to place the average accuracy marker
        for j, term in enumerate(scenario_results.keys()):
            term_result = next((res for res in scenario_results[term] if f"{parameter}={param_value}" in res), None)
            accuracy = float(term_result.split('accuracy=')[1]) if term_result else 0
            term_accuracies[term].append(accuracy)
            bar_pos = i - (total_width / 2) + (bar_width * j)
            plt.bar(bar_pos, accuracy, bar_width, color=colors[j], label=term if i == 0 else "")

        # Calculate the average accuracy for this parameter value
        all_accuracies = [acc for acc in term_accuracies.values()]
        avg_accuracy = np.mean(all_accuracies)
        avg_accuracies.append(avg_accuracy)

    # Plot the average accuracy line
    plt.plot(avg_accuracies, color='red', label='Average', marker='o')

    # Set the labels and title of the plot
    plt.xlabel(f"{parameter.capitalize()} values")
    plt.ylabel("Accuracy")
    plt.title(f"{scenario_name} over varied {parameter}")

    # Place the x-axis labels in the center of the group of bars
    plt.xticks(np.arange(len(parameter_values)), parameter_values)

    plt.yticks(np.arange(0, 1.1, 0.05))  # Adjust the range and increment as needed
    plt.grid(True, which='both', axis='y', linestyle='--', linewidth=0.5)

    # Add grid, legend, and show the plot
    # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()  # Ensure everything fits without overlap
    plt.show()


def plot_accuracies(accuracies_dict, title):
    plt.figure(figsize=(10, 6))
    terms = list(accuracies_dict.keys())
    accuracies = list(accuracies_dict.values())

    # Convert terms to numeric values for the purpose of plotting the line of best fit
    numeric_terms = np.arange(len(terms))

    # Calculate the coefficients of the line of best fit
    coefficients = np.polyfit(numeric_terms, accuracies, 1)
    polynomial = np.poly1d(coefficients)

    # Generate y-values for the line of best fit
    trendline = polynomial(numeric_terms)

    plt.bar(terms, accuracies, color='skyblue')
    plt.plot(terms, trendline, color='red', label='Line of Best Fit')

    plt.xlabel('WSD Terms')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)  # Assuming accuracy is between 0 and 1
    plt.legend()
    plt.show()

def average_accuracy(accuracy_dict):
    total_accuracy = sum(accuracy_dict.values())
    return total_accuracy / len(accuracy_dict)

def filter_test_instances_by_term(test_instances, term):
    return {id: instance for id, instance in test_instances.items() if instance.lemma == term}


def preprocess_and_vectorize_test_data(test_instances, vectorizer, lemmatizer=None, stop_words=None):
    preprocessed_test_sentences = [preprocess_sentencetrain(" ".join(instance.context), lemmatizer, stop_words) for instance in test_instances.values()]
    X_test = vectorizer.transform(preprocessed_test_sentences)
    return X_test


def convert_synset_to_lemma_sense_key(synset):
    # Find the synset object
    synset_obj = wn.synset(synset)

    # Get the lemmas of the synset
    lemmas = synset_obj.lemmas()

    if lemmas:
        # We can take any lemma; here, we choose the first
        lemma = lemmas[0]

        # Construct and return the sense key
        sense_key = lemma.key()
        return sense_key
    else:
        return None

def evaluate_classifier(classifier, label_encoder, X_test, test_key, test_instances):
    # Predict the sense keys
    predictions = classifier.predict(X_test)
    converted_predictions = label_encoder.inverse_transform(predictions)

    # Convert predicted synset numbers to sense keys
    predicted_sense_keys = []
    for id, pred in zip(sorted(test_key), converted_predictions):
        sense_number = determine_sense_number(pred)
        predicted_sense_keys.append(synset_number_to_sense_key(pred, test_instances[id].lemma, sense_number))

    # Extract the actual sense keys from the test_key in order
    actual_sense_keys = [test_key[id][0] for id in sorted(test_key)]

    # Calculate accuracy
    accuracy = accuracy_score(actual_sense_keys, predicted_sense_keys)

    for i, (prediction, actual) in enumerate(zip(predicted_sense_keys, actual_sense_keys)):

        word = test_instances[sorted(test_key)[i]].lemma

        context = " ".join(test_instances[sorted(test_key)[i]].context)

        print(f"Word: {word}")

        print(f"Context: {context}")

        print(f"Prediction: {prediction}")

        print(f"Correct Answer: {actual}")

        print("\n")


    return accuracy

def determine_sense_number(pred):
    # Split the prediction string and extract the last part which is the sense number
    parts = pred.split('.')
    if len(parts) >= 3:
        # The last part is expected to be the sense number
        sense_number_str = parts[-1]

        try:
            # Convert the sense number string to an integer
            return int(sense_number_str)
        except ValueError:
            # Handle the case where the sense number is not an integer
            return None
    else:
        # If the prediction string does not conform to the expected format
        return None



def synset_number_to_sense_key(synset_number, lemma, sense_number):
    synset_types = {1: 'n', 2: 'v', 3: 'a', 4: 'r', 5: 's'}
    synset_pos = synset_types.get(synset_number, 'n')  # Default to noun if not found

    synsets = wn.synsets(lemma, pos=synset_pos)
    if synsets and len(synsets) >= sense_number:
        target_synset = synsets[sense_number - 1]
        return target_synset.lemmas()[0].key()

    # If not found, search all synsets of the given POS
    target_sense_key_suffix = f"{synset_pos}:{str(sense_number).zfill(2)}"
    for synset in wn.all_synsets(synset_pos):
        for synset_lemma in synset.lemmas():
            if synset_lemma.key().endswith(target_sense_key_suffix):
                return synset_lemma.key()

    print("ERROR IN synset_number_to_sense_key CONVERSION FUNCTION!")
    exit()



def add_to_seed_set(seed_set, term, lemmatizer, stop_words):
    additional_entries = []

    if term == "game":
        additional_entries = sense_sentences_dict_game
    elif term == "year":
        additional_entries = sense_sentences_dict_year
    elif term == "country":
        additional_entries = sense_sentences_dict_country
    elif term == "action":
        additional_entries = sense_sentences_dict_action
    elif term == "claim":
        additional_entries = sense_sentences_dict_claim
    elif term == "support":
        additional_entries = sense_sentences_dict_support

    elif term == "law":
        additional_entries = sense_sentences_dict_law

    elif term == "rule":
        additional_entries = sense_sentences_dict_rule

    elif term == "end":

        additional_entries = sense_sentences_dict_end



    else:
        print("WARNING: No additional entries for term '{}'".format(term))
        return seed_set

    for sense_key, sentences in additional_entries.items():
        for sentence in sentences:
            words = sentence.split()
            features = {'sentence': ' '.join(words)}  # Again, creating a single string

            seed_set.append((features, sense_key))

    return seed_set
def preprocess_sentencetrain(sentence, lemmatizer=None, stop_words=None):
    words = sentence.split()
    processed_words = []
    for word in words:
        if lemmatizer is not None:
            word = lemmatizer.lemmatize(word)
        if stop_words is None or word.lower() not in stop_words:
            processed_words.append(word)
    return ' '.join(processed_words)

def process_sentence(seed_set, lemmatizer=None, stop_words=None):
    processed_seed_set = []

    for (feature_dict, sense_key) in seed_set:
        processed_feature_dict = {}

        for key, value in feature_dict.items():
            # Process each word in the sentence
            processed_words = []
            for word in value.split():
                # Lemmatize if lemmatizer is provided
                if lemmatizer is not None:
                    word = lemmatizer.lemmatize(word)

                # Add the word if stop words removal is not required or if the word is not a stop word
                if stop_words is None or word.lower() not in stop_words:
                    processed_words.append(word)

            # Update the feature dictionary
            processed_feature_dict[key] = ' '.join(processed_words)

        processed_seed_set.append((processed_feature_dict, sense_key))

    return processed_seed_set


def feature_extraction_bigrams(seed_set):
    # Extract sentences from the seed set
    sentences = [feature_dict['sentence'] for feature_dict, _ in seed_set]

    # Initialize CountVectorizer to use bigrams
    vectorizer = CountVectorizer(ngram_range=(2, 2))

    # Fit and transform the sentences to bigram frequency matrix
    X = vectorizer.fit_transform(sentences)

    # Optionally, convert to dense format (if needed)
    # X_dense = X.toarray()

    # Get the feature names (bigrams)
    feature_names = vectorizer.get_feature_names_out()

    return X, feature_names

def filter_seed_set_by_sense_key(seed_set, filter_key):
    # Filter the seed set to include only entries where the sense key starts with the specified filter_key
    filtered_seed_set = [(feature_dict, sense_key) for feature_dict, sense_key in seed_set if sense_key.startswith(filter_key)]
    return filtered_seed_set

def bootstrap_algorithm_for_term(term, train_instances, lemmatizer=None, stop_words=None , confidence_threshold=0.3, num_iterations=30):
    seed_set = create_seed_set(term, lemmatizer, stop_words)
    seed_set = add_to_seed_set(seed_set, term, lemmatizer, stop_words)
    seed_set = filter_seed_set_by_sense_key(seed_set, term)
    #processed_seed_set= process_sentence(seed_set, lemmatizer, stop_words=None)

    vectorizer = CountVectorizer(ngram_range=(1, 1))
    classifier = SVC(probability=True)
    label_encoder = LabelEncoder()


    seed_set_size = len(seed_set)
    excluded_indices = set()
    iteration=0
    while iteration<num_iterations:
        #print(f"\nIteration {iteration + 1}")
        iteration += 1



        # Process and extract bigram features from seed set
        processed_seed_set = process_sentence(seed_set, lemmatizer, stop_words)
        sentences = [feature_dict['sentence'] for feature_dict, _ in processed_seed_set]
        sense_keys = [sense_key for _, sense_key in processed_seed_set]

        # Fit vectorizer and transform sentences to bigram counts
        X_train = vectorizer.fit_transform(sentences)
        y_train = label_encoder.fit_transform(sense_keys)

        # Train the classifier
        classifier.fit(X_train, y_train)

        # Prepare entire dataset for bigram transformation
        #all_sentences = [" ".join(instance.context) for instance in train_instances]
        preprocessed_all_sentences = [preprocess_sentencetrain(" ".join(instance.context), lemmatizer, stop_words) for i, instance in enumerate(train_instances) if i not in excluded_indices]
        X_all = vectorizer.transform(preprocessed_all_sentences)


        # Apply the classifier to the transformed dataset
        predictions = classifier.predict(X_all)
        probabilities = classifier.predict_proba(X_all)
        high_confidence_seed_set = []
        # Select instances with confidence above the threshold
        # Mapping from reduced indices to original indices
        index_mapping = [i for i in range(len(train_instances)) if i not in excluded_indices]

        high_confidence_indices = [i for i, p in enumerate(probabilities.max(axis=1)) if p >= confidence_threshold]

        for hc_index in high_confidence_indices:
            # Map the reduced index back to the original index
            actual_index = index_mapping[hc_index]
            context = train_instances[actual_index].context
            sense = predictions[hc_index]
            new_entry = (dict(sentence=" ".join(context)), label_encoder.inverse_transform([sense])[0])
            seed_set.append(new_entry)
            high_confidence_seed_set.append(new_entry)
            excluded_indices.add(actual_index)

        # Print the size of the seed set after the iteration
       # print(f"Seed Set Size After Iteration {iteration}: {len(seed_set)}")

        if len(excluded_indices) == len(train_instances):
            #print("All training instances have been included in the seed set.")
            break

    # The final model
    return classifier, vectorizer, label_encoder






#so, after the seed set, the dataset is lowercased and preprocessed. I should do this before (before chosing seed set)

#07216494845360824
#BEST ONE SO FAR
def feature_extraction_indi(instance, lemmatizer= None, stop_words = None):
    features = {}
    # Focus on the target word (specified by instance.index in the WSDInstance class)
    target_word = instance.context[instance.index]

    for i, word in enumerate(instance.context):
        # Process word replacements and lemmatization
        word = word.replace("-", " ").replace("--", " ").replace(" ", "_")
        if lemmatizer:
            word = lemmatizer.lemmatize(word).lower()
        if stop_words and word.lower() in stop_words:
            continue

        # Feature format: context_word_{relative_position}
        relative_position = i - instance.index  # Relative position of the word to the target word
        if word == target_word:
            features['target_word'] = word
        else:
            features['context_word_' + str(relative_position)] = word

    return features




def get_most_frequent_sense(word):
    """
    Get the most frequent sense of a word from WordNet
    """
    synsets = wn.synsets(word)
    if not synsets:
        return None
    return synsets[0]



def main():
    dev_instances, test_instances, dev_key, test_key, lemmatizer, stop_words = setup_and_load_data()



    print(find_top_words_with_senses(test_key, 5))

    #test_sense_key_conversion(dev_key)
    # Get seed words with known dominant WSD
    #seed_words = get_most_frequent_lemmas(dev_instances, dev_key)
    wsd_terms = get_most_frequent_lemmas(dev_instances, test_instances)
    print("Most frequent lemmas with known WSD:", wsd_terms)

    wsd_terms.append("support")
    wsd_terms.append("law")
    wsd_terms.append("rule")
    wsd_terms.append("end")


    train_data_folder= 'SemCor'
    training_data_wsd_term_0= load_and_filter_training_data(wsd_terms[0], train_data_folder)
    training_data_wsd_term_1= load_and_filter_training_data(wsd_terms[1], train_data_folder)
    training_data_wsd_term_2= load_and_filter_training_data(wsd_terms[2], train_data_folder)
    training_data_wsd_term_3= load_and_filter_training_data(wsd_terms[3], train_data_folder)
    training_data_wsd_term_4= load_and_filter_training_data(wsd_terms[4], train_data_folder)

    training_data_wsd_term_5 = load_and_filter_training_data(wsd_terms[5], train_data_folder)
    training_data_wsd_term_6 = load_and_filter_training_data(wsd_terms[6], train_data_folder)
    training_data_wsd_term_7 = load_and_filter_training_data(wsd_terms[7], train_data_folder)
    training_data_wsd_term_8 = load_and_filter_training_data(wsd_terms[8], train_data_folder)

    seed_word_counts_term_0 = count_seed_words(training_data_wsd_term_0)
    seed_word_counts_term_1= count_seed_words(training_data_wsd_term_1)
    seed_word_counts_term_2= count_seed_words(training_data_wsd_term_2)
    seed_word_counts_term_3= count_seed_words(training_data_wsd_term_3)
    seed_word_counts_term_4= count_seed_words(training_data_wsd_term_4)
    seed_word_counts_term_5= count_seed_words(training_data_wsd_term_5)
    seed_word_counts_term_6= count_seed_words(training_data_wsd_term_6)
    seed_word_counts_term_7= count_seed_words(training_data_wsd_term_7)
    seed_word_counts_term_8= count_seed_words(training_data_wsd_term_8)

    # Display the counts
    for lemma, count in seed_word_counts_term_0.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_1.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_2.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_3.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts

    for lemma, count in seed_word_counts_term_4.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    for lemma, count in seed_word_counts_term_5.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_6.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_7.items():
        print(f"Lemma '{lemma}': {count} occurrences")
    # Display the counts
    for lemma, count in seed_word_counts_term_8.items():
        print(f"Lemma '{lemma}': {count} occurrences")

    print("WSD Terms: " , wsd_terms)


    classifier0, vectorizer0, labelencoder0 = bootstrap_algorithm_for_term(wsd_terms[0], training_data_wsd_term_0, lemmatizer, stop_words)
    classifier1, vectorizer1, labelencoder1 = bootstrap_algorithm_for_term(wsd_terms[1], training_data_wsd_term_1, lemmatizer, stop_words)
    classifier2, vectorizer2, labelencoder2 = bootstrap_algorithm_for_term(wsd_terms[2], training_data_wsd_term_2, lemmatizer, stop_words)
    classifier3, vectorizer3, labelencoder3 = bootstrap_algorithm_for_term(wsd_terms[3], training_data_wsd_term_3, lemmatizer, stop_words)
    classifier4, vectorizer4, labelencoder4 = bootstrap_algorithm_for_term(wsd_terms[4], training_data_wsd_term_4, lemmatizer, stop_words)
    classifier5, vectorizer5, labelencoder5 = bootstrap_algorithm_for_term(wsd_terms[5], training_data_wsd_term_5, lemmatizer, stop_words)
    classifier6, vectorizer6, labelencoder6 = bootstrap_algorithm_for_term(wsd_terms[6], training_data_wsd_term_6, lemmatizer, stop_words)
    classifier7, vectorizer7, labelencoder7 = bootstrap_algorithm_for_term(wsd_terms[7], training_data_wsd_term_7, lemmatizer, stop_words)
    classifier8, vectorizer8, labelencoder8 = bootstrap_algorithm_for_term(wsd_terms[8], training_data_wsd_term_8, lemmatizer, stop_words)


    classifiers = [classifier0, classifier1, classifier2, classifier3,  classifier4, classifier5, classifier6,classifier7 ,classifier8 ]
    vectorizers = [vectorizer0, vectorizer1, vectorizer2, vectorizer3, vectorizer4,vectorizer5, vectorizer6, vectorizer7, vectorizer8 ]
    label_encoders = [labelencoder0, labelencoder1, labelencoder2, labelencoder3, labelencoder4,labelencoder5, labelencoder6, labelencoder7, labelencoder8 ]

    accuracies_no_preprocessing = {}
    accuracies_lemmatization_stop_words = {}
    accuracies_stop_words_only = {}
    accuracies_lemmatization_only = {}

    print("\nAccuracies for lemmatization + stop word removal:")
    # Lists to store accuracies


    for i, (classifier, vectorizer, label_encoder) in enumerate(zip(classifiers, vectorizers, label_encoders)):
        # Filter test instances for the current wsd_term
        filtered_test_instances = filter_test_instances_by_term(test_instances, wsd_terms[i])

        # Preprocess and vectorize filtered test data
        X_test = preprocess_and_vectorize_test_data(filtered_test_instances, vectorizer, lemmatizer, stop_words)

        # Filter the test_key accordingly
        filtered_test_key = {id: test_key[id] for id in filtered_test_instances}

        # Evaluate classifier
        accuracy = evaluate_classifier(classifier, label_encoder, X_test, filtered_test_key, filtered_test_instances)
        print(f"Accuracy for classifier {i} (term '{wsd_terms[i]}'): {accuracy}")
        accuracies_lemmatization_stop_words[wsd_terms[i]] = accuracy

    classifier0, vectorizer0, labelencoder0 = bootstrap_algorithm_for_term(wsd_terms[0], training_data_wsd_term_0,  stop_words= stop_words)
    classifier1, vectorizer1, labelencoder1 = bootstrap_algorithm_for_term(wsd_terms[1], training_data_wsd_term_1, stop_words= stop_words)
    classifier2, vectorizer2, labelencoder2 = bootstrap_algorithm_for_term(wsd_terms[2], training_data_wsd_term_2, stop_words= stop_words)
    classifier3, vectorizer3, labelencoder3 = bootstrap_algorithm_for_term(wsd_terms[3], training_data_wsd_term_3, stop_words= stop_words)
    classifier4, vectorizer4, labelencoder4 = bootstrap_algorithm_for_term(wsd_terms[4], training_data_wsd_term_4, stop_words= stop_words)
    classifier5, vectorizer5, labelencoder5 = bootstrap_algorithm_for_term(wsd_terms[5], training_data_wsd_term_5, stop_words= stop_words)
    classifier6, vectorizer6, labelencoder6 = bootstrap_algorithm_for_term(wsd_terms[6], training_data_wsd_term_6, stop_words= stop_words)
    classifier7, vectorizer7, labelencoder7 = bootstrap_algorithm_for_term(wsd_terms[7], training_data_wsd_term_7, stop_words= stop_words)
    classifier8, vectorizer8, labelencoder8 = bootstrap_algorithm_for_term(wsd_terms[8], training_data_wsd_term_8, stop_words= stop_words)

    classifiers = [classifier0, classifier1, classifier2, classifier3,  classifier4, classifier5, classifier6,classifier7 ,classifier8 ]
    vectorizers = [vectorizer0, vectorizer1, vectorizer2, vectorizer3, vectorizer4,vectorizer5, vectorizer6, vectorizer7, vectorizer8 ]
    label_encoders = [labelencoder0, labelencoder1, labelencoder2, labelencoder3, labelencoder4,labelencoder5, labelencoder6, labelencoder7, labelencoder8 ]

    print("\nAccuracies for stop word removal only:")

    for i, (classifier, vectorizer, label_encoder) in enumerate(zip(classifiers, vectorizers, label_encoders)):
        # Filter test instances for the current wsd_term
        filtered_test_instances = filter_test_instances_by_term(test_instances, wsd_terms[i])

        # Preprocess and vectorize filtered test data
        X_test = preprocess_and_vectorize_test_data(filtered_test_instances, vectorizer, stop_words=stop_words)

        # Filter the test_key accordingly
        filtered_test_key = {id: test_key[id] for id in filtered_test_instances}

        # Evaluate classifier
        accuracy = evaluate_classifier(classifier, label_encoder, X_test, filtered_test_key, filtered_test_instances)
        print(f"Accuracy for classifier {i} (term '{wsd_terms[i]}'): {accuracy}")
        accuracies_stop_words_only[wsd_terms[i]] = accuracy



    classifier0, vectorizer0, labelencoder0 = bootstrap_algorithm_for_term(wsd_terms[0], training_data_wsd_term_0,
                                                                           lemmatizer=lemmatizer)
    classifier1, vectorizer1, labelencoder1 = bootstrap_algorithm_for_term(wsd_terms[1], training_data_wsd_term_1,
                                                                           lemmatizer=lemmatizer)
    classifier2, vectorizer2, labelencoder2 = bootstrap_algorithm_for_term(wsd_terms[2], training_data_wsd_term_2,
                                                                           lemmatizer=lemmatizer)
    classifier3, vectorizer3, labelencoder3 = bootstrap_algorithm_for_term(wsd_terms[3], training_data_wsd_term_3,
                                                                           lemmatizer=lemmatizer)
    classifier4, vectorizer4, labelencoder4 = bootstrap_algorithm_for_term(wsd_terms[4], training_data_wsd_term_4,
                                                                           lemmatizer=lemmatizer)
    classifier5, vectorizer5, labelencoder5 = bootstrap_algorithm_for_term(wsd_terms[5], training_data_wsd_term_5,
                                                                           lemmatizer=lemmatizer)
    classifier6, vectorizer6, labelencoder6 = bootstrap_algorithm_for_term(wsd_terms[6], training_data_wsd_term_6,
                                                                           lemmatizer=lemmatizer)
    classifier7, vectorizer7, labelencoder7 = bootstrap_algorithm_for_term(wsd_terms[7], training_data_wsd_term_7,
                                                                           lemmatizer=lemmatizer)
    classifier8, vectorizer8, labelencoder8 = bootstrap_algorithm_for_term(wsd_terms[8], training_data_wsd_term_8,
                                                                           lemmatizer=lemmatizer)

    classifiers = [classifier0, classifier1, classifier2, classifier3, classifier4, classifier5, classifier6,
                   classifier7, classifier8]
    vectorizers = [vectorizer0, vectorizer1, vectorizer2, vectorizer3, vectorizer4, vectorizer5, vectorizer6,
                   vectorizer7, vectorizer8]
    label_encoders = [labelencoder0, labelencoder1, labelencoder2, labelencoder3, labelencoder4, labelencoder5,
                      labelencoder6, labelencoder7, labelencoder8]

    print("\nAccuracies for lemmatization only:")

    for i, (classifier, vectorizer, label_encoder) in enumerate(zip(classifiers, vectorizers, label_encoders)):

        # Filter test instances for the current wsd_term
        filtered_test_instances = filter_test_instances_by_term(test_instances, wsd_terms[i])

        X_test = preprocess_and_vectorize_test_data(filtered_test_instances, vectorizer, lemmatizer=lemmatizer)

        # Filter the test_key accordingly
        filtered_test_key = {id: test_key[id] for id in filtered_test_instances}

        # Evaluate classifier
        #there is definitley a bug in this evaluation schema, as it was wokring but now its not.
        accuracy = evaluate_classifier(classifier, label_encoder, X_test, filtered_test_key, filtered_test_instances)
        print(f"Accuracy for classifier {i} (term '{wsd_terms[i]}'): {accuracy}")
        accuracies_lemmatization_only[wsd_terms[i]] = accuracy

    classifier0, vectorizer0, labelencoder0 = bootstrap_algorithm_for_term(wsd_terms[0], training_data_wsd_term_0)
    classifier1, vectorizer1, labelencoder1 = bootstrap_algorithm_for_term(wsd_terms[1], training_data_wsd_term_1)
    classifier2, vectorizer2, labelencoder2 = bootstrap_algorithm_for_term(wsd_terms[2], training_data_wsd_term_2)
    classifier3, vectorizer3, labelencoder3 = bootstrap_algorithm_for_term(wsd_terms[3], training_data_wsd_term_3)
    classifier4, vectorizer4, labelencoder4 = bootstrap_algorithm_for_term(wsd_terms[4], training_data_wsd_term_4)
    classifier5, vectorizer5, labelencoder5 = bootstrap_algorithm_for_term(wsd_terms[5], training_data_wsd_term_5)
    classifier6, vectorizer6, labelencoder6 = bootstrap_algorithm_for_term(wsd_terms[6], training_data_wsd_term_6)
    classifier7, vectorizer7, labelencoder7 = bootstrap_algorithm_for_term(wsd_terms[7], training_data_wsd_term_7)
    classifier8, vectorizer8, labelencoder8 = bootstrap_algorithm_for_term(wsd_terms[8], training_data_wsd_term_8)


    classifiers = [classifier0, classifier1, classifier2, classifier3,  classifier4, classifier5, classifier6,classifier7 ,classifier8 ]
    vectorizers = [vectorizer0, vectorizer1, vectorizer2, vectorizer3, vectorizer4,vectorizer5, vectorizer6, vectorizer7, vectorizer8 ]
    label_encoders = [labelencoder0, labelencoder1, labelencoder2, labelencoder3, labelencoder4,labelencoder5, labelencoder6, labelencoder7, labelencoder8 ]

    print("\nAccuracies for no lemmatization nor stop word removal:")

    for i, (classifier, vectorizer, label_encoder) in enumerate(zip(classifiers, vectorizers, label_encoders)):
        # Filter test instances for the current wsd_term
        filtered_test_instances = filter_test_instances_by_term(test_instances, wsd_terms[i])

        # Preprocess and vectorize filtered test data
        X_test = preprocess_and_vectorize_test_data(filtered_test_instances, vectorizer)

        # Filter the test_key accordingly
        filtered_test_key = {id: test_key[id] for id in filtered_test_instances}

        # Evaluate classifier
        accuracy = evaluate_classifier(classifier, label_encoder, X_test, filtered_test_key, filtered_test_instances)
        print(f"Accuracy for classifier {i} (term '{wsd_terms[i]}'): {accuracy}")
        accuracies_no_preprocessing[wsd_terms[i]] = accuracy

#todo: the line of best fit doesnt really make sense- do you mean making it one big graph with one LOBF?
    plot_accuracies(accuracies_no_preprocessing, "Accuracies with No Lemmatization Nor Stop Word Removal")
    plot_accuracies(accuracies_lemmatization_stop_words, "Accuracies with Lemmatization + Stop Word Removal")
    plot_accuracies(accuracies_stop_words_only, "Accuracies with Stop Word Removal Only")
    plot_accuracies(accuracies_lemmatization_only, "Accuracies with Lemmatization Only")

    #computing the average accuracies of each dictionary
    avg_accuracy_no_preprocessing = average_accuracy(accuracies_no_preprocessing)
    avg_accuracy_lemmatization_stop_words = average_accuracy(accuracies_lemmatization_stop_words)
    avg_accuracy_stop_words_only = average_accuracy(accuracies_stop_words_only)
    avg_accuracy_lemmatization_only = average_accuracy(accuracies_lemmatization_only)

    average_accuracies = {
        "No Preprocessing": avg_accuracy_no_preprocessing,
        "Lemmatization + Stop Words": avg_accuracy_lemmatization_stop_words,
        "Stop Words Only": avg_accuracy_stop_words_only,
        "Lemmatization Only": avg_accuracy_lemmatization_only
    }

    highest_avg_scenario = max(average_accuracies, key=average_accuracies.get)
    lowest_avg_scenario = min(average_accuracies, key=average_accuracies.get)

    print("\n Averages:")
    print("  Highest avegage pre-processing accuracy:" , highest_avg_scenario)
    print("  Lowest avegage pre-processing accuracy:" , lowest_avg_scenario)

    accuracy_dicts = [
        accuracies_no_preprocessing,
        accuracies_lemmatization_stop_words,
        accuracies_stop_words_only,
        accuracies_lemmatization_only
    ]
    total_accuracies = {}

    for acc_dict in accuracy_dicts:
        for term, accuracy in acc_dict.items():
            if term not in total_accuracies:
                total_accuracies[term] = []
            total_accuracies[term].append(accuracy)

    for term, accuracies in total_accuracies.items():
        print(f"Term: {term}, Accuracies: {accuracies}")

    term_avg_accuracies = {term: mean(accuracies) for term, accuracies in total_accuracies.items()}
    lowest_avg_accuracy_term = min(term_avg_accuracies, key=term_avg_accuracies.get)

    evaluate_scenarios(highest_avg_scenario, lowest_avg_scenario,test_key, test_instances,lemmatizer,stop_words,  wsd_terms,
                       train_data_folder)







if __name__ == '__main__':
    main()

