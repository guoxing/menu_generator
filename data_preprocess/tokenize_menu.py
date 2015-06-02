from stanford_corenlp_pywrapper import sockwrap
import pickle, re

'''
Tokenize menu data.

Input file: buzs_simple.pickle
Output file: buzs_tokenized.pickle
    contains a dictionary from buz_id -> business.
'''

jars = ['/Users/guoxing/code/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2.jar',
        '/Users/guoxing/code/stanford-corenlp-full-2015-04-20/stanford-corenlp-3.5.2-models.jar']

p = sockwrap.SockWrap("ssplit",corenlp_jars=jars, configdict={'annotators': 'tokenize, ssplit'})

BUZ_FILE_NAME = 'buzs_simple.pickle'
TOKENIZED_FILE_NAME = 'buzs_tokenized.pickle'

buzs = pickle.load(open(BUZ_FILE_NAME))

# tokenize menus
count = 0
for buz_id, buz in buzs.items():
    count += 1
    menu = []
    for food in buz['menu']:
        if food['type'] != 'ITEM' or 'name' not in food:
            continue
        result = p.parse_doc(food['name'])
        if len(result['sentences']) > 1:
            print '=========== Error ============'
            print result['sentences']
            print food['name']
        tokens = []
        for sentence in result['sentences']:
            tokens += [re.sub('\d', 'D', token) for token in sentence['tokens']]
        food['name'] = tokens
        menu.append(food)
    buz['menu'] = menu
    if count % 500 == 0:
        print 'count:', count
        print buz['menu'][0]

pickle.dump(buzs, open(TOKENIZED_FILE_NAME, 'w'))
