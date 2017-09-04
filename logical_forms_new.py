
'''
All functions involved in executing logical forms (obtained by parsing sentences) 
Feel free to add more functions if there is a need, don't be afraid to have to many - as anyway
we use a subset of these functions, that is defined in a different file.   
'''

from typing import *
from collections import Iterable
from structured_rep_utils import *




class Location(Enum):
    TOP = 'top'
    SECOND = 'second'
    BOTTOM = 'bottom'

class Relation(Enum):
    ABOVE = 'above'
    BELOW = 'below'
    TOUCH = 'touch'
    CLOSELY_TOUCH = 'closely touch'

class Side(Enum):
    RIGHT = 'right',
    LEFT = 'left',
    TOP = 'top',
    BOTTOM = 'bottom'
    ANY = 'any'


def exist(_set : set):
    return count(_set) > 0


def unique(_set : set):
    if len(_set)==1:
        return list(_set)[0]
    raise ValueError


def count(_set):
    return len(set([id(x) for x in _set]))

###################
### boolean item functions
####################

# color

def is_yellow(x):
    return x.color == Color.YELLOW
def is_blue(x):
    return x.color == Color.BLUE
def is_black(x):
    return x.color == Color.BLACK

# shape
def is_circle(x):
    return x.shape == Shape.CIRCLE
def is_square(x):
    return x.shape == Shape.SQUARE
def is_triangle(x):
    return x.shape == Shape.TRIANGLE

#size
def is_big(x):
    return x.size == Size.BIG
def is_medium(x):
    return x.size == Size.MEDIUM
def is_small(x):
    return x.size == Size.SMALL

# relative location

def is_top(x):
    return x.is_top()
def is_bottom(x):
    return x.is_bottom()
def is_second(x):
    return x.is_second()
def is_third(x):
    return x.is_third()

    # right and left not used for now

# touching wall


def is_touching_wall(x, side=None):
    if side == Side.ANY or side is None:
        return x.touching_wall()
    if side == Side.TOP:
        return x.touching_top()
    if side == Side.BOTTOM:
        return x.touching_bottom()
    if side == Side.RIGHT:
        return x.touching_right()
    if side == Side.LEFT:
        return x.touching_left()


def is_closely_touching_wall(x, side=None):
    if side == Side.ANY or side is None:
        return x.touching_wall(use_margin=True)
    if side == Side.TOP:
        return x.touching_top(use_margin=True)
    if side == Side.BOTTOM:
        return x.touching_bottom(use_margin=True)
    if side == Side.RIGHT:
        return x.touching_right(use_margin=True)
    if side == Side.LEFT:
        return x.touching_left(use_margin=True)

def is_touching_corner(x, side = None):
    return x.touching_corner() and is_touching_wall(x, side)


####################
### query functions for item properties
####################


def __set_per_item_function(s, f):
    if type(s) is Item:
        return [f(s)]
    if isinstance(s, Iterable) and all([type(x)==Item for x in s]):
        return [f(x) for x in s]
    else:
        raise TypeError("s is neither an item nor a set of items")

def query_color(x):
    return __set_per_item_function(x, lambda item : item.color)

def query_size(x):
    return __set_per_item_function(x, lambda item: item.size)

def query_shape(x):
    return __set_per_item_function(x, lambda item: item.shape)




############
##filters
############

def filter(_set, func):
    return [x for x in _set if func(x)]

## specific filters: we can do without them but they might be helpful in case we need
## to avoid too many lambda expressions

def filter_color(_set, color : Color):
     return filter(_set, lambda x : equal(query_color(x), color))

def filter_size(_set, size : Size):
     return filter(_set, lambda x : equal(query_size(x), size))

def filter_shape(_set, shape : Shape):
     return filter(_set, lambda x : equal(query_shape(x), shape))



##################
# comparison functions
#################


# integer comparison


def le(a,b):
    if not (type(a) is int and type(b) is int):
        raise TypeError
    return a<= b

def ge(a,b):
    if not (type(a) is int and type(b) is int):
        raise TypeError
    return a>= b

def lt(a,b):
    if not (type(a) is int and type(b) is int):
        raise TypeError
    return a< b

def gt(a,b):
    if not (type(a) is int and type(b) is int):
        raise TypeError
    return a> b

def equal_int(a,b):
    if not (type(a) is int and type(b) is int):
        raise TypeError
    return a == b

# generic equality check. note that singletons are defined here to be equal to their sole item,
# e.g. equal(Color.Blue, {Color.Blue}) is True

def equal(a,b):
    if type(a) == type(b):
        return a == b
    a = a if isinstance(a, list) else [a]
    b = b if isinstance(b, list) else [b]
    types = [type(x) for x in a] + [type(x) for x in b]
    if count(types) > 1:
        raise TypeError("cannot equate types: {0} and {1}".format(types[0], types[1]))
    return equal_set(a, b)


# type-specific equality functions. we can decide whether to use them or the generic equal.


def equal_color(a, b):
    if not (__is_type_or_set_of_type(a, Color) and __is_type_or_set_of_type(b, Color)):
        raise TypeError
    return equal(a, b)


def equal_size(a, b):
    if not (__is_type_or_set_of_type(a, Size) and __is_type_or_set_of_type(b, Size)):
        raise TypeError
    return equal(a, b)


def equal_shape(a, b):
    if not (__is_type_or_set_of_type(a, Shape) and __is_type_or_set_of_type(b, Shape)):
        raise TypeError
    return equal(a, b)


## relation functions - given an Item or a set of items, return a set pf items in
## a specific relation with it/them.


def get_above(s):
    return union_all(__set_per_item_function(s, lambda x : relate(Relation.ABOVE, x)))

def get_below(s):
    return union_all(__set_per_item_function(s, lambda x : relate(Relation.BELOW, x)))


def get_touching(s):
    return union_all(__set_per_item_function(s, lambda x : relate(Relation.TOUCH, x)))

def get_closely_touching(s):
    return union_all(__set_per_item_function(s, lambda x : relate(Relation.CLOSELY_TOUCH, x)))

def relate(rel:Relation, item):
    return set([x for x in get_box_exclusive(item) if __check_relation(x,item,rel)])


def __check_relation(x,item,rel):
    if rel== Relation.ABOVE:
        return item.top <= x.bottom and (item.is_touching(x) if item.box.is_tower() else True)
    if rel== Relation.BELOW:
        return __check_relation(item, x, Relation.ABOVE)
    if rel == Relation.TOUCH:
        return item.is_touching(x)
    if rel == Relation.CLOSELY_TOUCH:
        return item.is_touching(x, use_margin= True)
    raise TypeError("{} is not a relation".format(rel))


def get_box_inclusive(item : Item):
    return [x for x in item.box]


def get_box_exclusive(item : Item):
    return [x for x in item.box if x is not item]


# Logical operators

def AND(a,b):
    if type(a) is not bool or type(b) is not bool:
        raise TypeError
    return a and b

def OR(a,b):
    if type(a) is not bool or type(b) is not bool:
        raise TypeError
    return a or b

def NOT(a):
    if type(a) is not bool:
        raise TypeError
    return not(a)


## other functions

def All(_set, func):
    return all([func(x) is True for x in _set]) if len(_set)>0 else False


def Any(_set, func):
    for x in _set:
        if func(x) is True:
            return True
    return False


def member_of(item: Item, _set):
    if item in _set:
        return True
    return False


def contained(set1 , set2):
    for item in set1:
        if item not in set2:
            return False
    return True


def equal_set(set1, set2):
    if contained(set1,set2) and contained(set2, set1):
        return True
    return False


def all_same(_set):
    return len(set(_set)) == 1


def all_same_attribute(_set, func):
     return all_same([func(x) for x in _set])


def all_same_shape(_set):
    return all_same(query_shape(_set))


def all_same_color(_set):
    return all_same(_set, lambda x: query_color(x))


def union_all(sets):
    return set([item for subset in sets for item in subset])

def union(set1, set2):
    return set1.union(set2)

def intersect(set1, set2):
    return set1.intersection(set2)

def intersect_all(sets):
    l = list(sets)
    if count(l)==0:
        return set()
    return [x for x in l[0] if all([x in s for s in sets])]



def select_integers(k, min, max):
    if k> max-min or k==0:
        return [set()]
    return [set([min]).union(s) for s in select_integers(k-1, min+1, max)] + [s for s in select_integers(k, min+1, max) if len(s)>0]

def __is_type_or_set_of_type(x, t : type):
    return isinstance(x, t) or (isinstance(x, Iterable) and all([isinstance(a, t) for a in x]))

def select(k, _set):
    l = list(_set)
    return [[l[i] for i in idx] for idx in select_integers(k, 0, len(l))]


###############
### utility methods
###############



def process_token_sequence(token_seq, tokens_mapping):
    '''
    
    :param token_seq:   a sequence of valid tokens in the formal language 
    :param tokens_mapping: a dictionary mapping tokens to the number of arguments they take
    :return: the sequence with added brackets and commas,  runnable by python's eval function
    '''
    stack = []

    token_seq = token_seq.copy()

    for idx, token in enumerate(token_seq):
        n_args = len(tokens_mapping[token].args_types) if token in tokens_mapping else 0

        next_ch = ''
        added_chars = ""
        if token.startswith('lambda'):
            token = token.replace('_', ' ')


        elif n_args == 0:
            while next_ch != ', ' and len(stack)>0:
                next_ch = stack.pop()
                added_chars += next_ch
        elif n_args == 1:
            added_chars += '('
            stack.append(')')
        elif n_args == 2:
            added_chars += '('
            stack.extend([')',', '])
        token += added_chars
        token_seq[idx] = token

    return " ".join(token_seq)



def run_logical_form(expression, image):
    '''
    
    :param expression: a logical form (string)
    :param image: an object of type Image (a strutured representation of an image)
    :return: the result of executing the logical form on the structured representation
    '''
    # create constants
    all_boxes = image.get_all_boxes()
    all_items = image.get_all_items()


    f = eval("lambda ALL_BOXES, ALL_ITEMS : " + expression)
    result = f(all_boxes, all_items) # no error handling here for now...


    if type(result) is not bool:
        raise TypeError("parsing returned a non boolean type")
    return result


def parse(sentence, pre_parsed_dic = {}):
    # a mock: as we don't have a parser yet, the parse just returns a parsing from a dictionary of pre-parsed sentences
    if sentence in pre_parsed_dic:
        return pre_parsed_dic[sentence][1]
    else:
        raise NotImplementedError


