# Prerequisites

* Python **3.11.4**
  * `virtualenv` python package
    * to install, run: `pip install virtualenv`
* pip **24.0**
* **Windows 10** or above

# Setup

Navigate to the **project's root**.

Create a virtual environment:
```shell
virtualenv .venv
```

Activate the environment:
```shell
.venv/Scripts/activate
```

### Install dependencies

```shell
pip install pygame
pip install torch torchvision
pip install matplotlib ipython
pip install pyyaml
```

#### Explanation:

* `pygame`: visualization, player controls
* `torch`, `torchvision`: machine learning
* `matplotlib` `ipython`: plots
* `pyyaml`: yaml imports

To deactivate the environment:
```shell
deactivate
```

# Run

Activate the `.venv` and run the following command from the **project's root**:

```shell
python src/game.py
```

# Features

lorem ipsum

## New car

To add a **new car**, follow these steps:

### Image

Add your image (png) to the `src/resources/cars` folder.

* recommended size: ~80x40 pixels.

> **NOTE**: The image of the car must be facing to the right.

> **NOTE**: The name of the image will be used as the `car_id` in the following steps.

### yaml

Add the car and its attributes to the `cars.yaml` file in the `src/resources` folder.

The values are percentage based (between the `min_values` and `max_values`).

* `0` will set it according to the `min_values`.
* `100` will set it according to the `max_values`.
* anything outside this range may give unrealistic or unexpected behaviour.

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

> **NOTE**: Import the enums to access the cars: `from utils.enums import CarID`.

## New track

To add a **new track**, follow these steps:

### Image

Add your image (png) to the `src/resources/tracks` folder.

* recommended size: the same as the game window (see `game.yaml` in the `src/resources` folder).

> **NOTE**: The name of the image will be used as the `track_id` in the following steps.

### yaml

Add the track and its attributes to the `tracks.yaml` file in the `src/resources` folder.

#### Format:

```yaml
<track_id>:
  size: <num>
  car_default_state:
    x_position: <num>
    y_position: <num>
    angle: <num>
  checkpoints:
    0:
      x: <num>
      y: <num>
    1:
      x: <num>
      y: <num>
#   ...
```

Fill the values inside the `<>` brackets.

* `size`: the size multiplier of the car. Changes the size of the image, and the attributes accordingly.
  * default: `1`
* `car_default_state`: the default position of the car when the game is reset.
* `checkpoints`: at least 2 checkpoints are required

### Enum

Go to the `src/utils/enums.py` file and add a new value according to your `track_id`.

#### Format: `EXAMPLE = "track_id"`

### Usage

When creating a new track, you can use the enumerator.

#### Example: `example_track = Track(TrackID.EXAMPLE.value)`

 > **NOTE**: Import the enums to access the tracks: `from utils.enums import TrackID`.

## Tests

### Static Code Analysis

The `pylint` python package is required.

#### Install

Run the following command inside the `.venv`:

```shell
pip install pylint
```

#### Run

Run the following command from the project's root:

```shell
pylint src
```
