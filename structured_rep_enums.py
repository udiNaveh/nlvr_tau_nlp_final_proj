from enum import Enum


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