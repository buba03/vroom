""" Module containing various Enum classes. """

from enum import Enum


class Color(Enum):
    """
    Enumeration for RGB colors.
    When using .value, returns the tuple containing the RGB values for a color.
    """
    WHITE = (255, 255, 255)
    RED = (200, 0, 0)
    BLACK = (0, 0, 0)
    GRASS = (20, 130, 30)
    TRACK = (189, 190, 192)
    DEBUG = (0, 0, 255)


class Direction(Enum):
    """
    Enumeration for directions.
    """
    LEFT = 0
    RIGHT = 1
    UP = 3
    DOWN = 4
