import os
import pygame

from utils.yaml_manager import YamlManager


class Track:
    def __init__(self, track_id):
        # Set ID
        self.id = track_id

        # Set car image according to the car_id
        self.image = self._set_image(os.path.join('resources', 'tracks', track_id + '.png'))

        # yaml import
        track_attributes = YamlManager(os.path.join('resources', 'tracks.yaml')).get_track_attributes(track_id)

        # Set values from yaml
        self.car_default_state = track_attributes['car_default_state']
        self.car_size = track_attributes['size']

    @staticmethod
    def _set_image(path: str):
        """
        Load an image and scale it.

        :param path: Path to the image.
        :return: The image.
        """
        img = pygame.image.load(path)
        # TODO fix hardcoded size
        return pygame.transform.scale(img, (900, 600))

    def get_car_default_state(self):
        return (self.car_default_state['x_position'],
                self.car_default_state['y_position'],
                self.car_default_state['angle'])

    def draw(self, surface):
        surface.blit(self.image, (0, 0))
