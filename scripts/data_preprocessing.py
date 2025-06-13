import re

# Load the KJV Bible text
with open('data/kjv_1611.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Simple preprocessing: remove non-alphanumeric characters, convert to lowercase
text = re.sub(r'[^a-zA-Z\s]', '', text).lower()

# Split text into words
words = text.split()

print(f"Total words in KJV Bible: {len(words)}")

# Save preprocessed text if needed
with open('data/kjv_1611_preprocessed.txt', 'w', encoding='utf-8') as file:
    file.write(' '.join(words))
