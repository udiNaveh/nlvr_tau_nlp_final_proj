import requests
from requests.auth import HTTPDigestAuth

def increment_count(count_dict, key):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += 1
    else:
        count_dict[key] = 1


def call_api(word):
    url = 'https://wordsapiv1.p.mashape.com/words/' + word
    params = {    "X-Mashape-Key": "Z2LBQaaPOHmshmma0G7uyyxGP0nhp1XEXg2jsnq3bdFVtpXMa5",
    "Accept": "application/json"}
    myResponse = requests.get(url, headers = params)
    if myResponse.status_code != 200:
        return None
    return myResponse.json()