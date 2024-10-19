import yaml


class YamlManager:
    def __init__(self, path):
        self.path = path
        self.values = self.load_yaml(self.path)

    @staticmethod
    def load_yaml(path):
        with open(path, 'r') as file:
            return yaml.safe_load(file)

    def get_car_attributes(self, car_id):
        normalized_attributes = {}

        for attribute in self.values[car_id]:
            # (max - min) * (value / 100) + min
            max_value = self.values['max_values'][attribute]
            min_value = self.values['min_values'][attribute]
            value = self.values[car_id][attribute]

            normalized_attributes[attribute] = (max_value - min_value) * (value / 100) + min_value

        return normalized_attributes
