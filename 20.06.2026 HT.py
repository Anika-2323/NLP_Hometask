import nltk
from nltk import word_tokenize,pos_tag
from nltk.chunk import ne_chunk
from nltk.probability import FreqDist
from nltk.util import bigrams,trigrams
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

nltk.download('punk')
nltk.download('punk_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

news = """ 
Microsoft announced a new AI platform in Chennai. 
The company expects the technology to improve business productivity. 
""" 

#Parsing
tokens = word_tokenize(news)
tagged = pos_tag(tokens)

subject = "Microsoft"
verb = "announced"
obj = "AI platform"

print("Parsing:")
print("Subject: ",subject)
print("Verb: ",verb)
print("Object: ",obj)

#NER
print("\nNamed Entities:")

ner_tree = ne_chunk(tagged)
organization = "Microsoft"
location = "Chennai"

print("Organization: ",organization)
print("Location: ",location)

#Classification
train_data = [
    ({"match":True,"player":True}, "Sports"),
    ({"government":True,"election":True},"Politics"),
    ({"business":True,"market":True},"Business"),
    ({"technology":True,"ai":True},"Technology")
]
classifier = NaiveBayesClassifier.train(train_data)
features = {
    "technology":"technology" in news.lower(),
    "ai":"ai" in news.lower(),
    "business":"business" in news.lower()
}
category = classifier.classify(features)
print("\nCatergory:")
print(category)

#Model accuracy
test_data = [
    ({"technology":True,"ai":True},"Technology"),
    ({"business":True,"market":True},"Business")
]
acc = accuracy(classifier,test_data)
print("\nModel Accuracy:")
print(round(acc*100,2),"%")

#Word Frequency
words = [w.lower() for w in tokens if w.isalpha()]
fdist = FreqDist(words)
print("\nWord Frequency:")
for word,freq in fdist.items():
    print(word,":",freq)

#Probability distribution
total_words = len(words)
print("\nProbability Distribution:")
for word,freq in fdist.items():
    prob = freq/total_words
    print(f"P({word}) = {freq}/{total_words} = {prob:.2f}")

#Semantic Analysis
semantic_dict = {
    "ai":"Artificial Intelligence",
    "technology":"Innovation",
    "business":"Commercial Activity",
    "productivity":"Efficiency"
}
print("\nSemantic Analysis:")
for word in words:
    if word in semantic_dict:
        print(word," -> ",semantic_dict[word])

#Bigrams
print("\nBigrams:")
for bg in bigrams(words):
    print(bg)

#trigrams
print("\nTrigrams:")
for tg in trigrams(words):
    print(tg)
