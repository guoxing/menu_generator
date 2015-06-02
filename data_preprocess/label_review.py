import pickle, json

'''
Label business reviews using BILOU.
Each token in review are labeled as tuple.
('Fried', 'B'), ('Rice', 'L')
Each line in the resulting json file contains a json-encoded review

Input file: buzs_tokenized.pickle
            tokenized_reviews.pickle
Output file: labeled_reviews.json
'''

BUZ_FILE_NAME = 'buzs_tokenized.pickle'
REVIEW_FILE_NAME = 'tokenized_reviews.pickle'
REVIEW_LABEL_FILE_NAME = 'labeled_reviews.json'

buzs = pickle.load(open(BUZ_FILE_NAME))
buz_reviews = pickle.load(open(REVIEW_FILE_NAME))

def is_match(start_idx, review_tokens, food_tokens):
    if start_idx + len(food_tokens) > len(review_tokens):
        return False
    match = True
    for idx in range(len(food_tokens)):
        if review_tokens[start_idx + idx].lower() != food_tokens[idx].lower():
            match = False
            break
    return match

review_count = 0
matched_food_count = 0
with open(REVIEW_LABEL_FILE_NAME, 'w') as f:
    for buz_id, reviews in buz_reviews.items():
        menu = buzs[buz_id]['menu']
        for review in reviews:
            review_count += 1
            if review_count % 10000 == 0:
                print 'review count:', review_count
                print 'matched food count:', matched_food_count
            labeled_tokens = []
            tokens = review['text']
            idx = 0
            while idx < len(tokens):
                if tokens[idx] == '\n':
                    # special label for end of line
                    labeled_tokens.append((tokens[idx], '-'))
                    idx += 1
                    continue
                # check match
                match = False
                for food in menu:
                    food_tokens = food['name']
                    if len(food_tokens) < 1:
                        continue
                    if is_match(idx, tokens, food_tokens):
                        match = True
                        matched_food_count += 1
                        food_len = len(food_tokens)
                        if food_len == 1:
                            labeled_tokens.append((tokens[idx], 'U'))
                        else:
                            print 'food_len:', food_len
                            # food_len >= 2
                            labeled_tokens.append((tokens[idx], 'B'))
                            for i in range(1, food_len - 1):
                                labeled_tokens.append((tokens[idx + i], 'I'))
                            labeled_tokens.append((tokens[idx + food_len - 1], 'L'))
                        idx += food_len
                        break
                if not match:
                    labeled_tokens.append((tokens[idx], 'O'))
                    idx += 1
            review['text'] = labeled_tokens
            f.write(json.dumps(review) + '\n')


print 'final review count:', review_count
print 'final matched food count:', matched_food_count
