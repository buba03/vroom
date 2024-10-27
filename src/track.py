""" Module for racetrack implementation. """

import os
import pygame

from utils.yaml_manager import YamlManager


def set_image(path: str):
    """
    Load an image and scale it.

    :param path: Path to the image.
    :return: The image.
    """
    img = pygame.image.load(path)
    # TODO fix hardcoded size
    return pygame.transform.scale(img, (900, 600))


class Track:
    """ Represents a racetrack for a car. """

    def __init__(self, track_id):
        """
        Sets up the track object according to the track_id.

        :param track_id: The name of the track inside the tracks.yaml file.
        """
        # Set ID
        self.id = track_id

        # Set track image according to the track_id
        self.image = set_image(os.path.join('resources', 'tracks', track_id + '.png'))

        # yaml import
        track_attributes = YamlManager(os.path.join('resources', 'tracks.yaml')).get_track_attributes(track_id)

        # Set values from yaml
        self.car_default_state = track_attributes['car_default_state']
        self.car_size = track_attributes['size']

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
