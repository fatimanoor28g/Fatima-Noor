
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

text = """
Natural Language Processing is interesting.
Computers can process text and understand patterns.
Natural language processing is used in machine learning.
"""

# Sentence tokenization
sentences = sent_tokenize(text)

# Word tokenization
words = word_tokenize(text.lower())

# Remove punctuation
words = [word for word in words if word.isalpha()]

# Remove common stopwords
stop_words = set(stopwords.words("english"))
filtered_words = [
    word for word in words if word not in stop_words
]

# Apply stemming
stemmer = PorterStemmer()
stemmed_words = [
    stemmer.stem(word) for word in filtered_words
]

print("Original Text:")
print(text)

print("\nSentences:")
print(sentences)

print("\nAll Words:")
print(words)

print("\nWords after Stopword Removal:")
print(filtered_words)

print("\nStemmed Words:")
print(stemmed_words)
