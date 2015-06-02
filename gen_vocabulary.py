import pickle
from collections import defaultdict

'''
Generate vocabulary along with its frequency.
Words are lower-cased.
The result is sorted based on frequency.

Input file: tokenized_reviews.pickle
Output file: vocabulary.txt
                each line of which contains a word
'''

REVIEW_FILE_NAME = 'tokenized_reviews.pickle'
VOCABULARY_FILE_NAME = 'vocabulary.txt'

buz_reviews = pickle.load(open(REVIEW_FILE_NAME))

vocab = defaultdict(int)

for buz_id, reviews in buz_reviews.items():
    for review in reviews:
        for token in review['text']:
            vocab[token.lower().encode('ascii', 'ignore')] += 1

result = sorted(vocab.items(), key=lambda x: x[1], reverse=True)

with open(VOCABULARY_FILE_NAME, 'w') as f:
    for word, freq in result:
        f.write(word + ' ' + str(freq) + '\n')
