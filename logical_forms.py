
'''
All functions involved in executing logical forms (obtained by parsing sentences) 
on structured representations (Image objects) should be here
'''


from structured_rep_utils import *


class Location(Enum):
    TOP = 'top'
    SECOND = 'second'
    BOTTOM = 'bottom'


def exist(_set):
    return count(_set) > 0


def count(_set):
    return len(_set)


def filter(func, _set):
    return [x for x in _set if func(x)]


def filter_color(color : Color, _set):
    return filter(lambda x : equal_color(x.color, color), _set)


def equal_color(color1 : Color, color2 :Color):
    return color1 == color2


def le(a,b):
    return a<= b

def ge(a,b):
    return a>= b

def lt(a,b):
    return a< b

def gt(a,b):
    return a> b

def equal_int(a,b):
    return a== b

def get_box_inclusive(item):
    return [x for x in item.box.get_all_items()]

def get_box_exclusive(item):
    return [x for x in item.box.get_all_items() if x!=item]

def filter_size(size : Size, _set):
    return filter(lambda x : equal_size(x.size, size), _set)

def equal_size(size1 : Size, size2 : Size):
    return size1 == size2


def filter_shape(shape : Shape, _set):
    return filter(lambda x : equal_shape(x.shape, shape), _set)


def equal_shape(shape1 : Shape, shape2 : Shape):
    return shape1 == shape2


def filter_location(loc : Location, _set):
    return filter(lambda x : equal_location(x, loc), _set)


def equal_location(x : Item, loc : Location):
# Comparison only for Items, not Boxes. However it is probably not needed for Boxes
    if x.is_top():
        if loc==Location.TOP:
            return True
        return False
    if x.is_bottom() :
        if loc==Location.BOTTOM:
            return True
        return False
    if loc==Location.SECOND:
        return True
    return False


def query_color(x : Item):
    return x.color


def query_size(x : Item):
    return x.size


def query_shape(x : Item):
    return x.shape


def query_location(x : Item):
    if x.is_top():
        return Location.TOP
    if x.is_bottom():
        return Location.BOTTOM
    return Location.SECOND


def query_touching_right_wall(x : Item):
    return x.touching_right()


def query_touching_left_wall(x : Item):
    return x.touching_left()


def query_touching_top_wall(x : Item):
    return x.touching_top()


def query_touching_bottom_wall(x : Item):
    return x.touching_bottom()


def query_touching_corner(x : Item):
    return x.touching_corner()


def query_touching_wall(x : Item):
    b = query_touching_right_wall or query_touching_left_wall or query_touching_top_wall or query_touching_bottom_wall
    return b

# AND, OR and NOT can be the regular functions just written using lowercase letters


def All(func, _set):
    for x in _set:
        if not func(x):
            return False
    return True


def Any(func, _set):
    for x in _set:
        if func(x):
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


def all_same_size(_set):
    myList=list(_set)
    first=myList[0]
    for item in myList:
        if item.size!=first.size:
            return False
    return True


def all_same_shape(_set):
    myList=list(_set)
    first=myList[0]
    for item in myList:
        if item.shape!=first.shape:
            return False
    return True


def all_same_color(_set):
    myList=list(_set)
    first=myList[0]
    for item in myList:
        if item.color!=first.color:
            return False
    return True
