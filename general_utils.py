import requests
from requests.auth import HTTPDigestAuth

def increment_count(count_dict, key, n=1):
    """
        Puts the key in the dictionary if does not exist or adds one if it does.
        Args:
            count_dict: a dictionary mapping a string to an integer
            key: a string
    """
    if key in count_dict:
        count_dict[key] += n
    else:
        count_dict[key] = n


def union_dicts(dict1, dict2):
    return dict(list(dict1.items()) + list(dict2.items()))


def union_count_dicts(dict1,dict2):
   all_keys = set(list(dict1.keys()) + list(dict2.keys()))
   return {k : dict1.get(k,0) + dict2.get(k,0) for k in all_keys}