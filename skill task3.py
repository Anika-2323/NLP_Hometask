
import re
from collections import Counter


# --------------------------------------------------
# 1. Load the Kaggle dataset
# --------------------------------------------------

filename = r"C:\Semester 5\NLP Skill\Class task\skill task3.txt"

with open(filename, "r", encoding="utf-8") as file:
    text = file.read()


# --------------------------------------------------
# 2. Clean and tokenize the text
# --------------------------------------------------

text = text.lower()

# Keep only alphabetic words
tokens = re.findall(r"\b[a-z]+\b", text)

print("Total Tokens:", len(tokens))


# --------------------------------------------------
# 3. Build Unigram, Bigram and Trigram tables
# --------------------------------------------------

# Unigram frequency
unigrams = Counter(tokens)

# Bigram frequency
bigrams = Counter(
    zip(tokens, tokens[1:])
)

# Trigram frequency
trigrams = Counter(
    zip(tokens, tokens[1:], tokens[2:])
)


print("Vocabulary Size:", len(unigrams))
print("Total Bigrams:", len(bigrams))
print("Total Trigrams:", len(trigrams))


# --------------------------------------------------
# 4. Calculate Unigram probabilities
# --------------------------------------------------

total_words = sum(unigrams.values())

unigram_probabilities = {
    word: count / total_words
    for word, count in unigrams.items()
}


# --------------------------------------------------
# 5. Predict Next Words
# --------------------------------------------------

def predict_next_words(sentence, top_n=5):

    words = re.findall(r"\b[a-z]+\b", sentence.lower())

    if not words:
        return []

    candidates = {}

    # ============================================
    # 1. TRIGRAM candidates
    # ============================================

    if len(words) >= 2:

        w1 = words[-2]
        w2 = words[-1]

        for (a, b, c), count in trigrams.items():

            if a == w1 and b == w2:

                context_count = bigrams[(w1, w2)]

                probability = count / context_count

                candidates[c] = probability


    # ============================================
    # 2. BIGRAM candidates
    # ============================================

    previous_word = words[-1]

    if len(candidates) < top_n:

        word_count = unigrams[previous_word]

        for (a, b), count in bigrams.items():

            if a == previous_word:

                probability = count / word_count

                if b not in candidates:
                    candidates[b] = probability


    # ============================================
    # 3. UNIGRAM candidates
    # ============================================

    if len(candidates) < top_n:

        for word, count in unigrams.most_common():

            if word not in candidates:

                probability = count / total_words

                candidates[word] = probability

            if len(candidates) >= top_n:
                break


    # ============================================
    # 4. Sort candidates
    # ============================================

    ranked = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked[:top_n]



# --------------------------------------------------
# 6. User Input
# --------------------------------------------------

print("\n" + "=" * 60)
print("SMART NEXT-WORD PREDICTOR")
print("=" * 60)

sentence = input(
    "\nEnter a sentence or partial sentence: "
)


# --------------------------------------------------
# 7. Top-N
# --------------------------------------------------

try:

    top_n = int(
        input("Enter number of predictions (3-5): ")
    )

    if top_n < 3:
        top_n = 3

    elif top_n > 5:
        top_n = 5

except ValueError:

    top_n = 5


# --------------------------------------------------
# 8. Generate Predictions
# --------------------------------------------------

predictions = predict_next_words(
    sentence,
    top_n
)


# --------------------------------------------------
# 9. Display Output
# --------------------------------------------------

print("\nOriginal Input:")
print(sentence)

print("\nTop Predicted Next Words:")

if predictions:

    for i, (word, probability) in enumerate(
        predictions, 1
    ):

        print(
            f"{i}. {word} - {probability:.3f}"
        )

else:

    print("No predictions found.")

print("=" * 60)
Total Tokens: 108882
Vocabulary Size: 8034
Total Bigrams: 52330
Total Trigrams: 92153

============================================================
SMART NEXT-WORD PREDICTOR
============================================================

Enter a sentence or partial sentence:  this is
Enter number of predictions (3-5):  3

Original Input:
this is

Top Predicted Next Words:
1. my - 0.175
2. a - 0.150
3. the - 0.150
============================================================
