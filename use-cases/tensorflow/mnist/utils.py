import keras
import tensorflow as tf
import json

def to_json(model: keras.Model, filepath: str):
    with open(filepath, "w") as f:
        json.dump(model.to_json(), f)

def from_json(filepath: str) -> keras.Model:
    with open(filepath, "r") as f:
        config = json.load(f)
        return tf.keras.models.model_from_json(config)
