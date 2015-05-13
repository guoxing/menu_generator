import pickle

'''
Extract menu info from raw data.

After running this script, buzs_simple.pickle should contain a dictionary
of buz_id -> buz, where buz has basic business info as well as menu data in
buz['menu']. buz['menu'] contains a list of food.

Input file: buzs_with_menu.pickle
Output file: buzs_simple.pickle
'''

buzs = pickle.load(open('buzs_with_menu.pickle'))
buzs_simple = {}

count = 0
for buz_id, buz in buzs.items():
    count += 1
    print count, buz_id
    if ('venues' not in buz) or (len(buz['venues']) == 0):
        print 'No venues for {}'.format(buz_id)
    venue = buz['venues'][0]
    if ('menus' not in venue) or (len(venue['menus']) == 0):
        print 'No menus for {}'.format(buz_id)
    menus = venue['menus']
    food = []
    for menu in menus:
        for section in menu['sections']:
            for subsection in section['subsections']:
                for content in subsection['contents']:
                    food.append(content)
    del buz['venues']
    buz['menu'] = food
    buzs_simple[buz_id] = buz

pickle.dump(buzs_simple, open('buzs_simple.pickle', 'w'))
