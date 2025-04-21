import matplotlib.pyplot as plt

from nltk.corpus import brown
from nltk.tokenize import RegexpTokenizer
from collections import Counter

import spacy

# Load English language model
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 6000000


### PART 1 ###


# Tokenize words (No digits or punctuation. Keep contractions and hyphenated words.)
pattern = r"\b[a-zA-Z]+(?:[-'][a-zA-Z]+)*\b"
regexp_tokenizer = RegexpTokenizer(pattern)

# Convert tokens to lowercase
def clean_tokens(words, tokenizer):
    return [token.lower() for token in tokenizer.tokenize(words)]

# Unique list of all words in descending order by count
corpus_words = ' '.join(brown.words())
corpus_tokens = clean_tokens(corpus_words, regexp_tokenizer)
corpus_count = Counter(corpus_tokens).most_common()

corpus_descending_list = []
for word, freq in corpus_count:
    corpus_descending_list.append(word)

# Unique list of 'lore' words in descending order by count
lore_words = ' '.join(brown.words(categories='lore'))
lore_tokens = clean_tokens(lore_words, regexp_tokenizer)
lore_count = Counter(lore_tokens).most_common()

lore_descending_list = []
for word, freq in lore_count:
    lore_descending_list.append(word)

# Unique list of 'humor' words in descending order by count
humor_words = ' '.join(brown.words(categories='humor'))
humor_tokens = clean_tokens(humor_words, regexp_tokenizer)
humor_count = Counter(humor_tokens).most_common()

humor_descending_list = []
for word, freq in humor_count:
    humor_descending_list.append(word)


### PART 2 ###


print('----PART 2----')
print()

# Corpus information (i-vi)
print('Corpus Information')
print(f"(i) Number of tokens: {len(corpus_tokens)}")
print(f"(ii) Number of types: {len(set(corpus_tokens))}")
print(f"(iii) Number of words: {len(corpus_tokens)}")
print(f"(iv) Average number of words per sentence: {len(corpus_tokens) / len(brown.sents()):.3f}")
print(f"(v) Average word length (v): {sum(len(word) for word in corpus_tokens) / len(corpus_tokens):.3f}")

corpus_text = ' '.join(corpus_tokens)
corpus_document = nlp(corpus_text)
lemmas = [token.lemma_.lower() for token in corpus_document if token.is_alpha]
lemma_count = len(set(lemmas))
print(f"(vi) Number of lemmas: {lemma_count}")
print()

# 'Lore' information (i-vi)
print('Lore Information')
print(f"(i) Number of tokens: {len(lore_tokens)}")
print(f"(ii) Number of types: {len(set(lore_tokens))}")
print(f"(iii) Number of words: {len(lore_tokens)}")
print(f"(iv) Average number of words per sentence: {len(lore_tokens) / len(brown.sents(categories='lore')):.3f}")
print(f"(v) Average word length (v): {sum(len(word) for word in lore_tokens) / len(lore_tokens):.3f}")

lore_text = ' '.join(lore_tokens)
lore_document = nlp(lore_text)
lemmas = [token.lemma_.lower() for token in lore_document if token.is_alpha]
lemma_count = len(set(lemmas))
print(f"(vi) Number of lemmas: {lemma_count}")
print()

# 'Humor' information (i-vi)
print('Humor Information')
print(f"(i) Number of tokens: {len(humor_tokens)}")
print(f"(ii) Number of types: {len(set(humor_tokens))}")
print(f"(iii) Number of words: {len(humor_tokens)}")
print(f"(iv) Average number of words per sentence: {len(humor_tokens) / len(brown.sents(categories='humor')):.3f}")
print(f"(v) Average word length (v): {sum(len(word) for word in humor_tokens) / len(humor_tokens):.3f}")

humor_text = ' '.join(humor_tokens)
humor_document = nlp(humor_text)
lemmas = [token.lemma_.lower() for token in humor_document if token.is_alpha]
lemma_count = len(set(lemmas))
print(f"(vi) Number of lemmas: {lemma_count}")
print()


### STEP 3 ###


print('----PART 3----')
print()

# Top 10 POS tags for corpus in descending order by count
print('Corpus POS Tags')
corpus_pos_tags = [token.pos_ for token in corpus_document if token.is_alpha]
corpus_pos_count = Counter(corpus_pos_tags).most_common(10)
corpus_top_tags = [tag for tag, freq in corpus_pos_count]
print(corpus_top_tags)
print()

# Top 10 POS tags for 'lore' in descending order by count
print('Lore POS Tags')
lore_pos_tags = [token.pos_ for token in lore_document if token.is_alpha]
lore_pos_count = Counter(lore_pos_tags).most_common(10)
lore_top_tags = [tag for tag, freq in lore_pos_count]
print(lore_top_tags)
print()

# Top 10 POS tags for 'humor' in descending order by count
print('Humor POS Tags')
humor_pos_tags = [token.pos_ for token in humor_document if token.is_alpha]
humor_pos_count = Counter(humor_pos_tags).most_common(10)
humor_top_tags = [tag for tag, freq in humor_pos_count]
print(humor_top_tags)
print()


### STEP 4 ###


# Establish order of words based on frequencies
corpus_frequencies = [freq for word, freq in corpus_count]
lore_frequencies = [freq for word, freq in lore_count]
humor_frequencies = [freq for word, freq in humor_count]

corpus_order = list(range(1, len(corpus_frequencies) + 1))
lore_order = list(range(1, len(lore_frequencies) + 1))
humor_order = list(range(1, len(humor_frequencies) + 1))

# Linear plot
plt.figure(figsize=(12, 8))
plt.plot(corpus_order, corpus_frequencies, label='Corpus')
plt.plot(lore_order, lore_frequencies, label='Lore')
plt.plot(humor_order, humor_frequencies, label='Humor')
plt.title('Word Frequencies (Linear)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Log plot
plt.figure(figsize=(12, 8))
plt.loglog(corpus_order, corpus_frequencies, label='Corpus')
plt.loglog(lore_order, lore_frequencies, label='Lore')
plt.loglog(humor_order, humor_frequencies, label='Humor')
plt.title('Word Frequencies (Log)')
plt.xlabel('Rank')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, which="both")
plt.tight_layout()
plt.show()














