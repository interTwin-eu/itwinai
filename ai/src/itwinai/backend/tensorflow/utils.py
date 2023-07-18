import keras
import json

def model_to_json(model: keras.Model, filepath: str):
    with open(filepath, "w") as f:
        json.dump(model.to_json(), f)

def model_from_json(filepath: str) -> keras.Model:
    with open(filepath, "r") as f:
        config = json.load(f)
        return keras.models.model_from_json(config)