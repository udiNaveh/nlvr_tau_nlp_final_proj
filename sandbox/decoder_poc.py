'''
decoder poc
'''

from structured_rep_utils import *
from numpy import random as random

class Decoder:


    def __init__(self):
        self.Sets = {
            "Color" : {'black', 'blue', 'yellow'},
            "Shape" : {'square', 'triangle', 'circle'},
            "Relation" : {'above', 'below'},
            "Item" : set(),
            "Set" : {"all_items" : "Item", "Boxes" : "Set"},
            "Int" : {1,2,3,4,5,6,7},
            #"Bool": {True, False},
            "Location": {"top", "bottom"}
        }

        self.Functions = {
            "count" : (["Set"] , "Int"),
            "exist" : (["Set"] , "Bool"),
            "relate" : (("Relation","Set"), "Set"),
            "equal_integer" : (("Int", "Int"), "Bool"),
            #"less_equal" : (("Int", "Int"), "Bool"),
            #"bigger_equal" : (("Int", "Int"), "Bool"),
            "all_same" : (["Set"], "Bool"),
            "and" : (("Bool", "Bool"), "Bool"),
            "or" : (("Bool", "Bool"), "Bool"),
            "not" : (["Bool"], "Bool"),
            "equal_set": (("Set", "Set"), "Bool"),
            "query_color": (["Item"], "Color"),
            "query_shape": (["Item"], "Shape"),
            "filter" : (("Set", "lambda"), "Set")
        }

        self.vars = [c for c in "abcdefghijklmnopqrstuvwxyz"]

        self.stack = ["Bool"]

        self.on_set = None

        self.decode = []

    def choose_next_token(self):
        if len(self.stack)==0:
            return "."
        next_type = self.stack[-1]
        literals = [str(l) for l in self.Sets.get(next_type, set())]
        funcs = [f[0] for f in self.Functions.items() if f[1][1] == next_type]
        both = literals + funcs
        self.stack.pop()
        if len(both)>0:
            idx = random.randint(0, len(both))
            next_token = both[idx]

            if idx < len(literals):
                if next_type=='Set':
                    self.on_set = self.Sets["Set"][next_token]
                else:
                    self.on_set = None
            else:
                arguments = self.Functions[next_token][0]
                self.stack.extend(arguments[::-1])


        elif next_type != "lambda":
            raise TypeError(next_type)

        else:
            var = self.vars.pop(0)
            next_token = "lambda " + var
            if self.on_set == "Set":
                self.Sets["Set"][var] = "Item"
            else:
                self.Sets[self.on_set].add(var)

        return next_token

    def get_decoding(self):
        result = []
        while True:
            result.append(self.choose_next_token())
            if result[-1]=='.':
                break
        return result
















def try_stuff():

    for _ in range(100):
        dec = Decoder()
        print (dec.get_decoding())



if __name__ == "__main__":
    try_stuff()

