import json


def load_json_from_path(path):
    '''
        Input: path to json file
        Return: A dictionary of the json file
    '''
    with open('output/retrain/frozen/options.json', 'r') as f:
        data = json.load(f)
    return data