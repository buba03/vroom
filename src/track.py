""" Module for racetrack implementation. """

import pygame

from utils.config_manager import ConfigManager


def set_image(path: str):
    """
    Load an image and scale it.

    :param path: Path to the image.
    :return: The image.
    """
    img = pygame.image.load(path).convert_alpha()
    # TODO fix hardcoded size
    return pygame.transform.scale(img, (900, 600))


def set_checkpoints(points: dict) -> list[tuple[float, float]]:
    """
    Make a list of the points from the dictionary in the yaml file.

    :param points: The dictionary of the checkpoints from the yaml file.
    :return: A list of tuples with the coordinates.
    """

    return [(point['x'], point['y']) for point in points.values()]


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

        # Set values from yaml
        self.car_size = track_attributes['size']
        self.car_default_state = track_attributes['car_default_state']
        self.checkpoints = set_checkpoints(track_attributes['checkpoints'])

        # Set track image according to the track_id
        self.image = set_image(ConfigManager().get_track_image_path(track_id))

    def get_car_default_state(self) -> tuple[float, float, float]:
        """
        Returns the recommended default state for the car.

        :return: The car's x, y position and the angle as a tuple.
        """
        return (self.car_default_state['x_position'],
                self.car_default_state['y_position'],
                self.car_default_state['angle'])

    def draw(self, surface):
        """
        Draws the track on the given surface.

        :param surface: The surface the track should be displayed on.
        """
        surface.blit(self.image, (0, 0))
