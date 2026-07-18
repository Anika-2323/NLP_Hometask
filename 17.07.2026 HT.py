import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ---------------- Load Dataset ----------------

data = pd.read_csv("pos_tags.csv", nrows=1000)

print(data.head())

# ---------------- Create Sentences ----------------

sentences = []

temp = []

for _, row in data.iterrows():

    temp.append((row["word"], row["tag"]))

    if len(temp) == 20:
        sentences.append(temp)
        temp = []

# ---------------- Train Test Split ----------------

train_data, test_data = train_test_split(
    sentences,
    test_size=0.2,
    random_state=42
)

# ---------------- Vocabulary ----------------

vocabulary = set()
tags = set()

for sentence in train_data:
    for word, tag in sentence:
        vocabulary.add(word.lower())
        tags.add(tag)

word_to_index = {word: i for i, word in enumerate(sorted(vocabulary))}
tag_to_index = {tag: i for i, tag in enumerate(sorted(tags))}
index_to_tag = {i: tag for tag, i in tag_to_index.items()}

V = len(vocabulary)
T = len(tags)

print("Vocabulary Size :", V)
print("Number of Tags :", T)

# ---------------- HMM Matrices ----------------

initial = np.ones(T)

transition = np.ones((T, T))

emission = np.ones((T, V))

# ---------------- Count Probabilities ----------------

for sentence in train_data:

    first_tag = sentence[0][1]

    initial[tag_to_index[first_tag]] += 1

    for i, (word, tag) in enumerate(sentence):

        word = word.lower()

        tag_index = tag_to_index[tag]

        if word in word_to_index:
            emission[tag_index, word_to_index[word]] += 1

        if i > 0:

            previous_tag = sentence[i - 1][1]

            transition[tag_to_index[previous_tag], tag_index] += 1

# ---------------- Normalize ----------------

initial /= initial.sum()

transition /= transition.sum(axis=1, keepdims=True)

emission /= emission.sum(axis=1, keepdims=True)

# ---------------- Log Probabilities ----------------

log_initial = np.log(initial)

log_transition = np.log(transition)

log_emission = np.log(emission)

# ---------------- Vectorized Viterbi ----------------

def viterbi(sentence):

    n = len(sentence)

    dp = np.zeros((T, n))

    backpointer = np.zeros((T, n), dtype=int)

    word = sentence[0].lower()

    if word in word_to_index:
        emit = log_emission[:, word_to_index[word]]
    else:
        emit = np.log(np.ones(T) * 1e-10)

    dp[:, 0] = log_initial + emit

    for i in range(1, n):

        word = sentence[i].lower()

        if word in word_to_index:
            emit = log_emission[:, word_to_index[word]]
        else:
            emit = np.log(np.ones(T) * 1e-10)

        scores = dp[:, i - 1][:, None] + log_transition

        backpointer[:, i] = np.argmax(scores, axis=0)

        dp[:, i] = np.max(scores, axis=0) + emit

    best = np.argmax(dp[:, -1])

    result = [best]

    for i in range(n - 1, 0, -1):
        best = backpointer[best, i]
        result.append(best)

    result.reverse()

    return [index_to_tag[i] for i in result]

# ---------------- Example Prediction ----------------

sentence = "Artificial intelligence improves healthcare systems".split()

prediction = viterbi(sentence)

print("\nPrediction")
print("----------------")

for word, tag in zip(sentence, prediction):
    print(word, "---->", tag)

# ---------------- Evaluation ----------------

actual = []

predicted = []

for sentence in test_data:

    words = [w for w, t in sentence]

    true = [t for w, t in sentence]

    pred = viterbi(words)

    actual.extend(true)

    predicted.extend(pred)

print("\nAccuracy :", accuracy_score(actual, predicted))

print("\nClassification Report\n")

print(classification_report(actual, predicted, zero_division=0))

# ---------------- Unseen Sentences ----------------

test_sentences = [

    "Machine learning is amazing",

    "The boy is playing football",

    "She likes reading books",

    "Python makes coding easier",

    "Dogs bark loudly"

]

print("\nPredictions on Unseen Sentences")
print("--------------------------------")

for sent in test_sentences:

    words = sent.split()

    tags = viterbi(words)

    print("\nSentence :", sent)

    for w, t in zip(words, tags):
        print(f"{w:12} -> {t}")
