"""
CITS4404 Group C1

DAOs for storing phrase pairs (question, answer) in Mongodb
"""
import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/chatbot")
db = client['chatbot']

class PhrasePair:
    def __init__(self, name=None, x=None, y=None, dictionary=None):
        self.dict = dict({
            'name': name,
            'x': x,
            'y': y
        })
        if dictionary is not None:
            self.dict = dictionary

def save_data(phrase_pairs):
    dicts = [phrase_pair.dict for phrase_pair in phrase_pairs]

    count = 0
    for dict in dicts:
        db.phrase_pairs.insert_one(dict, True)
        count += 1

    print(count)

def load_data(ids=None):
    collection = db.phrase_pairs

    if ids is None:
        phrase_pairs_dict = collection.find()

    phrase_pairs = [PhrasePair(dictionary=dict) for dict in phrase_pairs_dict]
    return sorted(phrase_pairs, key=lambda phrase_pair: phrase_pair.dict['_id'])
