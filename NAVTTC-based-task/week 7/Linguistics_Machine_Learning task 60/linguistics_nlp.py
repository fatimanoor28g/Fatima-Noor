
import re
from collections import Counter

# Sample text
text = """
Natural language processing helps computers understand language.
Machine learning helps computers learn patterns from language.
Natural language processing uses machine learning.
"""

# Convert text to lowercase
clean_text = text.lower()

# Extract words
words = re.findall(r"\b[a-z]+\b", clean_text)

# Count word frequencies
word_counts = Counter(words)

print("Original Text:")
print(text)

print("\nProcessed Words:")
print(words)

print("\nWord Frequencies:")
for word, count in word_counts.most_common():
    print(f"{word}: {count}")

print("\nTotal words:", len(words))
print("Unique words:", len(word_counts))
