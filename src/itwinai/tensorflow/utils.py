# --------------------------------------------------------------------------------------
# Part of the interTwin Project: https://www.intertwin.eu/
#
# Created by: Matteo Bunino
#
# Credit:
# - Matteo Bunino <matteo.bunino@cern.ch> - CERN
# --------------------------------------------------------------------------------------


import json

import keras


def model_to_json(model: keras.Model, filepath: str):
    """Serialize Keras model to JSON file.

    Args:
        model (keras.Model): Keras model.
        filepath (str): JSON file path.
    """
    with open(filepath, "w") as f:
        json.dump(model.to_json(), f)


def model_from_json(filepath: str) -> keras.Model:
    """Deserialize Keras model from JSON file.

    Args:
        filepath (str): JSON file path.

    Returns:
        keras.Model: loaded Keras model.
    """
    with open(filepath, "r") as f:
        config = json.load(f)
        return keras.models.model_from_json(config)
