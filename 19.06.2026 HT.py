import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.probability import FreqDist

# Downloads (run once)
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

resume_text = "Iam Anika. Iam currently pursuing BSc.cs with AI in SDNB through Face prep"

# Tokenization
tokens = word_tokenize(resume_text)

print("Tokens:")
print(tokens)

# Stopword Removal
stop_words = set(stopwords.words('english'))

filtered_words = [
    word.lower()
    for word in tokens
    if word.isalpha() and word.lower() not in stop_words
]

print("\nAfter Stopword Removal:")
print(filtered_words)

# Stemming
stemmer = PorterStemmer()
stemmer_words = [stemmer.stem(word) for word in filtered_words]

print("\nStemmed Words:")
print(stemmer_words)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

print("\nLemmatized Words:")
print(lemmatized_words)

# POS Tagging
pos_tags = nltk.pos_tag(filtered_words)

print("\nPOS Tags:")
print(pos_tags)

# Frequency Distribution
freq_dist = FreqDist(filtered_words)

print("\nFrequency Distribution:")
for word, freq in freq_dist.items():
    print(f"{word}: {freq}")
