import re
import pandas as pd
import numpy as np

# --------------------------------------------------
# 1. Load Birkbeck spelling error corpus
# --------------------------------------------------

file_path = "missp.dat"

with open(file_path, "r", encoding="utf-8", errors="ignore") as file:
    lines = file.read().splitlines()


# --------------------------------------------------
# 2. Create spelling correction dictionary
# --------------------------------------------------

correction_dict = {}

correct_word = None

for line in lines:

    line = line.strip()

    if not line:
        continue

    # $ indicates the correct spelling
    if line.startswith("$"):
        correct_word = line[1:].lower()

    else:
        # This is a misspelled word
        misspelled_word = line.lower()

        if correct_word:
            correction_dict[misspelled_word] = correct_word


print("Total spelling corrections:",
      len(correction_dict))


# --------------------------------------------------
# 3. Convert dictionary to DataFrame
# --------------------------------------------------

df = pd.DataFrame(
    correction_dict.items(),
    columns=["misspelled", "correct"]
)

print("\nFirst 10 entries:")
print(df.head(10))


# --------------------------------------------------
# 4. Spelling Corrector
# --------------------------------------------------

def correct_word(word):

    word_lower = word.lower()

    # Check whether word exists in corpus
    if word_lower in correction_dict:
        return correction_dict[word_lower]

    # Already correctly spelled
    return word


# --------------------------------------------------
# 5. Correct complete search query
# --------------------------------------------------

def correct_query(query):

    # Extract words
    words = re.findall(r"[A-Za-z]+", query)

    corrected_words = []

    for word in words:
        corrected_words.append(correct_word(word))

    return " ".join(corrected_words)


# --------------------------------------------------
# 6. Test individual words
# --------------------------------------------------

test_words = [
    "Ameraca",
    "Amercia",
    "Cambrige",
    "Canda",
    "Apirl",
    "Christain",
    "Decmber",
    "Febuary",
    "Firday",
    "Munday"
]

print("\nWord Correction:")
print("----------------")

for word in test_words:

    corrected = correct_word(word)

    print(word, "->", corrected)


# --------------------------------------------------
# 7. Search Query Spelling Corrector
# --------------------------------------------------

print("\nSearch Query Spelling Corrector")
print("--------------------------------")

query = input("Enter your search query: ")

corrected_query = correct_query(query)

print("Original Query :", query)
print("Corrected Query:", corrected_query)
