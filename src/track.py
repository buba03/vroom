""" Module for racetrack implementation. """

from random import randint
import pygame

from utils.config_manager import ConfigManager


def set_image(path: str, size: tuple) -> pygame.image:
    """
    Load an image and scale it.

    :param path: Path to the image.
    :param size: The width and height of the image as a tuple.
    :return: The scaled image.
    """
    img = pygame.image.load(path).convert_alpha()
    return pygame.transform.scale(img, size)


def order_checkpoints(points: dict, index: int) -> dict:
    """
    Orders the checkpoints from the given dictionary and index.

    :param points: The dictionary of the checkpoints.
    :param index: Index of the checkpoint which should be the starting one.
    :return: A new dictionary of the ordered checkpoints.
    """
    if index not in points:
        raise ValueError('Starting index not found in points.')

    ordered_keys = sorted(points.keys())
    start_index = ordered_keys.index(index)
    reordered_keys = ordered_keys[start_index:] + ordered_keys[:start_index]

    ordered_dict = {k: points[k] for k in reordered_keys}
    return ordered_dict


class Track:
    """ Represents a racetrack for a car. """

    def __init__(self, track_id):
        """
        Initializes the track object according to the track_id.

        :param track_id: The name of the track inside the tracks.yaml file.
        """
        # Set ID
        self.id = track_id

        # yaml import
        track_attributes = ConfigManager().get_track_attributes(track_id)
        width = ConfigManager().get_game_attributes()['display']['width']
        height = ConfigManager().get_game_attributes()['display']['height']
        self.size = width, height  # tuple

        # Set values from yaml
        self.car_size = track_attributes['size']
        self.__car_default_states = track_attributes['car_default_states']
        self.__checkpoints = track_attributes['checkpoints']

        # Set track image according to the track_id
        self.image = set_image(ConfigManager().get_track_image_path(track_id), self.size)

    def get_car_default_state(self, random: bool = True) -> tuple[float, float, float]:
        """
        Return a random car state from the yaml file and sets the correct order of the checkpoints based on that.

        :param random: Whether to randomly return the car state or choose the first one.
        :return: The car's x, y position and the angle as a tuple.
        """
        num_of_checkpoints = len(self.__checkpoints)
        if random:
            index = randint(0, num_of_checkpoints - 1)
        else:
            index = 0

        self.__checkpoints = order_checkpoints(self.__checkpoints, index)
        return (self.__car_default_states[index]['x'],
                self.__car_default_states[index]['y'],
                self.__car_default_states[index]['angle'])

    def get_checkpoints(self) -> list[tuple[float, float]]:
        """
        Make a list of the points from the checkpoints dictionary.

        :return: A list of tuples with the coordinates.
        """
        return [(point['x'], point['y']) for point in self.__checkpoints.values()]

    def draw(self, surface):
        """
        Draws the track on the given surface.

        :param surface: The surface the track should be displayed on.
        """
        surface.blit(self.image, (0, 0))
