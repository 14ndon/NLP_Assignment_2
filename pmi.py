import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('./results', exist_ok=True)

with open('./brown_100.txt', 'r') as file_:
    corpus = file_.read()

def write_freqs(list_: list, filename: str) -> None:
    with open(f'./results/{filename}.txt', 'w') as file_:
        file_.write('\n'.join([' '.join(x[0]) + ' | ' + str(x[1]) for x in list_]))

from collections import Counter, defaultdict # You may import more from collections if needed

class PMICalculator:
    def __init__(self, corpus: str):
        self.corpus = corpus
        self.words = corpus.split()
        self.word_counts = Counter(self.words)
        self.word_pairs = defaultdict(int)
        self.total_words = len(self.words)
        self.pmis = {}

    def calculate_word_pairs(self):
        i = 0
        while i < len(self.words) - 1:
            word1 = self.words[i]
            word2 = self.words[i + 1]

            # only consider words that appear at least 10 times
            if self.word_counts[word1] < 10 or self.word_counts[word2] < 10:
                i += 1
                continue

            # skip if the word is a period and the next word is not the start token
            if word2 == '.' and self.words[i + 2] != "</s>":
                i += 2
                continue

            # if the end of the pair is an end token, count the pair but move on to the next line 
            elif word2 == "</s>":
                self.word_pairs[(word1, word2)] += 1
                i += 2
                continue

            # count the pair
            self.word_pairs[(word1, word2)] += 1
            i += 1

    
    def find_most_common_pairs(self, top_n: int = 10):
        pairs = list(self.word_pairs.items())
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_n]
    
    def calculate_pmi(self):
        for word1, word2 in self.word_pairs:
            count_word1 = self.word_counts[word1]
            count_word2 = self.word_counts[word2]
            count_word1_word2 = self.word_pairs[(word1, word2)]
            
            try:
                pmi = np.log2(count_word1_word2 * self.total_words / (count_word1 * count_word2))
            except Exception as e:
                print(word1, word2)
                print(count_word1, count_word2, count_word1_word2)
                print(self.total_words)
                print(e)
                return 0.0
            self.pmis[(word1, word2)] = pmi
        return self.pmis
    
    def sorted_pmi_pairs(self, top_n = 10, reverse = True):
        pairs = list(self.pmis.items())
        pairs.sort(key=lambda x: x[1], reverse=reverse)
        return pairs[:top_n]
    

corpus = open('./brown_100.txt', 'r').read()
pmi_calculator = PMICalculator(corpus)
pmi_calculator.calculate_word_pairs()
# write_freqs(pmi_calculator.find_most_common_pairs(), 'most_common_pairs')
pmi_calculator.calculate_pmi()
write_freqs(pmi_calculator.sorted_pmi_pairs(10, True), 'highest_pmi_pairs')
write_freqs(pmi_calculator.sorted_pmi_pairs(10, False), 'lowest_pmi_pairs')