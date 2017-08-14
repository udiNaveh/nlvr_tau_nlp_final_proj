
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


def unique(_set):
    if len(_set)==1:
        return list(_set)[0]
    return None


def count(_set):
    return len(_set)


def filter(func, _set):
    return [x for x in _set if func(x)]


def filter_color(color : Color, _set):
    color=Color(color)
    return filter(lambda x : equal_color(x.color, color), _set)


def equal_color(color1 : Color, color2 :Color):
    if not color1 or not color2:
        return False
    color1 = Color(color1)
    color2 = Color(color2)
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
    size=Size(size)
    return filter(lambda x : equal_size(x.size, size), _set)

def equal_size(size1 : Size, size2 : Size):
    if not size1 or not size2:
        return False
    size1 = Size(size1)
    size2 = Size(size2)
    return size1 == size2


def filter_shape(shape : Shape, _set):
    shape=Shape(shape)
    return filter(lambda x : equal_shape(x.shape, shape), _set)


def equal_shape(shape1 : Shape, shape2 : Shape):
    if not shape1 or not shape2:
        return False
    shape1 = Shape(shape1)
    shape2 = Shape(shape2)
    return shape1 == shape2


def filter_location(loc : Location, _set):
    loc=Location(loc)
    if loc==Location.TOP:
        return [x for x in _set if x.is_top()]
    if loc==Location.BOTTOM:
        return [x for x in _set if x.is_bottom()]
    return []
    #return filter(lambda x : equal_location(query_location(x), loc), _set)

def filter_top(_set):
    return [x for x in _set if x.is_top()]

def filter_bottom(_set):
    return [x for x in _set if x.is_bottom()]

def equal_location(loc1 : Location, loc2 : Location):
    if not loc1 or not loc2:
        return False
    loc1=Location(loc1)
    loc2=Location(loc2)
    return loc1==loc2



def query_color(x : Item):
    if not x:
        return None
    return x.color


def query_size(x : Item):
    if not x:
        return None
    return x.size


def query_shape(x : Item):
    if not x:
        return None
    return x.shape


def query_location(x : Item):
    if not x:
        return None
    if x.is_top():
        return Location.TOP
    if x.is_bottom():
        return Location.BOTTOM
    return Location.SECOND


def query_touching_right_wall(x : Item):
    if not x:
        return None
    return x.touching_right()


def query_touching_left_wall(x : Item):
    if not x:
        return None
    return x.touching_left()


def query_touching_top_wall(x : Item):
    if not x:
        return None
    return x.touching_top()


def query_touching_bottom_wall(x : Item):
    if not x:
        return None
    return x.touching_bottom()


def query_touching_corner(x : Item):
    if not x:
        return None
    return x.touching_corner()


def query_touching_wall(x : Item):
    if not x:
        return None
    b = query_touching_right_wall(x) or query_touching_left_wall(x) or query_touching_top_wall(x) or query_touching_bottom_wall(x)
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

def query_tower(box: Box):
    return box.is_tower()