"""
CITS4404 Group C1

DAOs for storing models in Mongodb
"""

import pymongo

client = pymongo.MongoClient("mongodb://localhost:27017/chatbot")
db = client['chatbot']

class DataModel:
    def __init__(self, name=None, training_ids_enc=None, training_ids_dec=None,
                 testing_ids_enc=None, testing_ids_dec=None, vocab_enc=None,
                 vocab_dec=None, dictionary=None):
        self.dict = dict({
            'name': name,
            'training_ids_enc': training_ids_enc,
            'training_ids_dec': training_ids_dec,
            'testing_ids_enc': testing_ids_enc,
            'testing_ids_dec': testing_ids_dec,
            'vocab_enc': vocab_enc,
            'vocab_dec': vocab_dec
        })
        if dictionary is not None:
            self.dict = dictionary

def save_data(data_model):
    dicts = [model.dict for model in data_model]

    count = 0
    for dict in dicts:
        db.data_models.insert_one(dict, True)
        count += 1

    print(count)

def load_data(ids=None):
    collection = db.data_models

    if ids is None:
        data_models_dict = collection.find()

    data_models = [DataModel(dictionary=dict) for dict in data_models_dict]
    return sorted(data_models, key=lambda data_model: data_model.dict['_id'])
