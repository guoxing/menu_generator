from stanford_corenlp_pywrapper import sockwrap
import json, pickle, re

'''
Tokenize business reviews. Each review becomes a list of tokens. Sentences are separated
by '\n' token.

Input file: yelp_academic_dataset_review.json
            buzs_simple.pickle
Output file: tokenized_reviews.pickle
'''

jars = ['/Users/guoxing/code/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar',
        '/Users/guoxing/code/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2-models.jar']

p = sockwrap.SockWrap("ssplit",corenlp_jars=jars, configdict={'annotators': 'tokenize, ssplit'})

REVIEW_FILE_NAME = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'
BUZ_FILE_NAME = 'buzs_simple.pickle'

buzs = pickle.load(open(BUZ_FILE_NAME))
buz_ids = buzs.keys()

buz_reviews = {buz_id: [] for buz_id in buz_ids}

# tokenize reviews
total_count = 0
relevant_count = 0
with open(REVIEW_FILE_NAME) as f:
    for line in f:
        total_count += 1
        review = json.loads(line)
        buz_id = review['business_id']
        if buz_id in buz_ids:
            relevant_count += 1
            result = p.parse_doc(review['text'])
            tokens = []
            for sentence in result['sentences']:
                tokens += [re.sub('\d', 'D', token) for token in sentence['tokens']] + ['\n']
            review['text'] = tokens
            buz_reviews[buz_id].append(review)
        if total_count % 10000 == 0:
            print 'Total review processed:', total_count
            print 'Relevant review:', relevant_count

pickle.dump(buz_reviews, open('tokenized_reviews.pickle', 'w'))
