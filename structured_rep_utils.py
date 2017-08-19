import typing
from enum import Enum

almost_touching_margin = 5


Colors = ['Yellow', 'Black', 'Blue']
Shapes = ['circle', 'square', 'triangle']


class Size(Enum):
    SMALL = 10
    MEDIUM = 20
    BIG = 30


class Color(Enum):
    YELLOW = 'Yellow'
    BLACK = 'Black'
    BLUE = '#0099ff'


class Shape(Enum):
    CIRCLE = 'circle'
    SQUARE = 'square'
    TRIANGLE = 'triangle'





class Item:
    def __init__(self, dic):
        assert isinstance(dic, dict)
        self.__y_loc = dic['y_loc']
        self.__x_loc = dic['x_loc']
        self.color = Color(dic['color'])
        self.size = Size(dic['size'])
        self.shape = Shape(dic['type'])
        self.box = None # a pointer to the containing box (List of Items). Added when the box constructor is done.

    def __repr__(self):
        return "{0} {1} {2} at x: ({3}-{4}) y: ({5},{6})".format(self.size.name, self.color.name, self.shape.name, self.left, self.right, self.bottom, self.top).lower()

    @property
    def right(self):
        return self.__x_loc + self.size.value

    @property
    def top(self):
        return 100 - self.__y_loc

    @property
    def bottom(self):
        return 100 - (self.__y_loc + self.size.value)

    # note: in the original representation the value of y_loc is bigger when the item is closer to the bottom.
    # here it is changed in order to be more intuitive and match the regular notion of x,y axes.

    @property
    def left(self):
        return self.__x_loc


    def touching_right(self, use_margin = False):
        margin = almost_touching_margin if use_margin else 0
        return self.right >= 100 - margin

    def touching_left(self, use_margin = False):
        margin = almost_touching_margin if use_margin else 0
        return self.left <= margin

    def touching_bottom(self, use_margin = False):
        margin = almost_touching_margin if use_margin else 0
        return self.bottom <= margin

    def touching_top(self, use_margin = False):
        margin = almost_touching_margin if use_margin else 0
        return self.top >= 100 - margin

    def touching_wall(self, use_margin = False):
        return self.touching_right(use_margin) or self.touching_left(use_margin) \
               or self.touching_bottom(use_margin) or self.touching_top(use_margin)

    def touching_corner(self, use_margin = False):
        return sum([self.touching_right(use_margin), self.touching_left(use_margin),
                    self.touching_bottom(use_margin), self.touching_top(use_margin)])==2

    def __distance(self, other):
        # in test assert that always >= 0 as items never overlap
        if not isinstance(other, Item):
            raise TypeError
        if self is other:
            return 0
        return max(self.left - other.right,
               other.left - self.right,
               self.bottom - other.top,
               other.bottom - self.top)

    def is_touching(self, other, use_margin=False):
        margin = almost_touching_margin if use_margin else 1
        return self is not other and self.box is other.box and self.__distance(other) <= margin

    def is_top(self):
        return self.top == max(item.top for item in self.box)

    def is_bottom(self):
        return self.top == min(item.top for item in self.box)

    def is_second(self):
        result = self.box.is_tower() and self.bottom==21
        return result

    def is_third(self):
        return self.box.is_tower() and self.bottom==42



class Box:

    def __init__(self, items_as_dicts : typing.List[dict]):
        self.items = [Item(d) for d in items_as_dicts]
        for item in self.items:
            item.box = self

    def __repr__(self):
        return "Box({})".format(self.items)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, key):
        return self.items[key]

    def __iter__(self):
        for s in self.items:
            yield  s

    def __contains__(self, item):
        return item in self.items

    def is_tower(self):
        return all(s.shape == Shape.SQUARE for s in self.items) and \
               all(s.right == self.items[0].right for s in self.items)


class Image:

    def __init__(self, structured_rep : typing.List[typing.List[dict]]):
        self.boxes = [Box(items_as_dicts) for items_as_dicts in structured_rep]

    def __len__(self):
        return len(self.boxes)

    def __getitem__(self, key):
        return self.boxes[key]

    def __iter__(self):
        for s in self.boxes:
            yield s

    def get_all_boxes(self):
        return self.boxes

    def get_all_items(self):
        return [item for box in self.boxes for item in box]

    def is_tower(self):
        return all([b.is_tower() for b in self.boxes])


class Sample:

    def __init__(self, line):
        self.evals = line["evals"]
        self.identifier = line["identifier"]
        self.sentence = line["sentence"]
        self.label = line["label"] == 'true'
        self.structured_rep = Image(line["structured_rep"])



