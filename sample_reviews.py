import json

'''
Sample a few reviews from a given business_id
'''

BUZ_ID = 'Vhv2KNJVPYykv7IPQv0MRw'
SAMPLE_SIZE = 10
FILE_NAME = 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_review.json'

with open(FILE_NAME) as f:
    count = 0
    for line in f:
        review = json.loads(line)
        if review['business_id'] == BUZ_ID:
            print review
            count += 1
            if count > SAMPLE_SIZE:
                break
