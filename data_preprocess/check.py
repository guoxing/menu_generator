import json, os
import pickle
import requests

BUZ_PICKLE_FILE = 'buzs.pickle'
BUZ_MENU_PICKLE_FILE = 'buzs_with_menu.pickle'

def readBuzs():
    if os.path.isfile(BUZ_PICKLE_FILE):
        buzs = pickle.load(open(BUZ_PICKLE_FILE))
    else:
        buzs = {}
    if os.path.isfile(BUZ_MENU_PICKLE_FILE):
        buzs_with_menu = pickle.load(open(BUZ_MENU_PICKLE_FILE))
    else:
        buzs_with_menu = {}
    return (buzs, buzs_with_menu)

buzs, buzs_with_menu = readBuzs()
count = 0
with open('yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json') as f:
    for line in f:
        count += 1
        if count % 100 == 0:
            print 'Num of buzs:', len(buzs)
            print 'Num of buzs with menu:', len(buzs_with_menu)
        buz = json.loads(line)
        if 'Restaurants' not in buz['categories']:
            continue
        if buz['business_id'] in buzs:
            continue
        buzs[buz['business_id']] = buz
        rq = {'fields':['name', 'menus'],
            'venue_queries':[{'name':buz['name'],
                                'location':{'locality':buz['city'], 'region':buz['state']},
                                'menus':{'$present':True}}],
            'api_key':'af31648d127d362397e9ad94c8df49665111f52c'}
        r = requests.post('https://api.locu.com/v2/venue/search/', json=rq)
        if r.status_code != 200:
            print '\033[91m' + 'Status code error:', r.status_code
            del buzs[buz['business_id']]
            break
        content = json.loads(r.content)
        if len(content['venues']) == 0:
            continue
        print "New buz_with_menu count:", count
        buz['venues'] = content['venues']
        buzs_with_menu[buz['business_id']] = buz


pickle.dump(buzs, open('buzs.pickle', 'w'))
pickle.dump(buzs_with_menu, open('buzs_with_menu.pickle', 'w'))
print 'Final result'
print 'Num of buzs:', len(buzs)
print 'Num of buzs with menu:', len(buzs_with_menu)
