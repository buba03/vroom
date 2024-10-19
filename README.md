# Prerequisites

* python: `3.11.4`
* pip: `24.0`
* ...

# Setup

Navigate to the project's root.

## Virtual environment

Install `virtualenv`:
```shell
?????
```

Create a virtual environment:
```shell
virtualenv .venv
```

Activate the environment:
```shell
.venv\Scripts\activate
```

### Install dependencies

```shell
pip install pygame
pip install torch torchvision
pip install matplotlib ipython
pip install pyyaml
```

To deactivate the environment, run:
```shell
deactivate
```

# Features

lorem ipsum

## New car

To add a new car, follow these steps:

### Image

Add your image (png) to the `resources` folder.

* recommended size: ~80x40 pixels.

**NOTE**: The image of the car must be facing to the right.

**NOTE**: The name of the image will be used as the `car_id` in the following steps..

### yaml

Add the car and its attributes to the `cars.yaml` file in the `reasources` folder.

#### Format:

```yaml
<car_id>:
  acceleration: <num>
  handling: <num>
  max_speed: <num>
  max_reverse_speed: <num>
  friction: <num>
```
 Fill the values inside the `<>` brackets.

### Enum

Go to the `src/utils/enums.py` file and add a new value according to your `car_id`.

#### Format: `EXAMPLE = "car_id"`

### Usage

When creating a new car, you can use the enumerator.

#### Example: `example_car = Car(CarID.EXAMPLE.value)`


**NOTE**: Import the enums to access the cars: `from utils.enums import CarID`.