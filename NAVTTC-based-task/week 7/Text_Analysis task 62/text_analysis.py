
import re
from collections import Counter

text = """
I love learning artificial intelligence.
Machine learning is useful and interesting.
I love Python because Python makes learning easier.
Sometimes debugging is difficult, but solving problems feels great.
"""

# Convert text to lowercase
clean_text = text.lower()

# Extract words
words = re.findall(r"\b[a-z]+\b", clean_text)

# Count word frequencies
word_counts = Counter(words)

# Simple example sentiment word lists
positive_words = {
    "love", "useful", "interesting", "easier", "great"
}
negative_words = {
    "difficult", "bad", "hate", "poor"
}

positive_count = sum(
    1 for word in words if word in positive_words
)
negative_count = sum(
    1 for word in words if word in negative_words
)

print("Text Analysis")
print("-------------")
print("Total words:", len(words))
print("Unique words:", len(set(words)))

print("\nMost Common Words:")
for word, count in word_counts.most_common(5):
    print(f"{word}: {count}")

print("\nPositive word count:", positive_count)
print("Negative word count:", negative_count)
