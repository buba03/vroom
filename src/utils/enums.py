from enum import Enum


class Color(Enum):
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLACK = (0, 0, 0)


class Direction(Enum):
    LEFT = 0
    RIGHT = 1


class CarID(Enum):
    FERRARI = "ferrari"
    MCLAREN = "mclaren"
