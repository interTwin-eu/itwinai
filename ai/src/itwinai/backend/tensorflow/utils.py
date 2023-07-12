import yaml
import keras
import json

#Parse (part of) YAML loaded in memory
def parse_pipe_config(yaml_file, parser):
    with open(yaml_file, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return parser.parse_object(config)

def model_to_json(model: keras.Model, filepath: str):
    with open(filepath, "w") as f:
        json.dump(model.to_json(), f)

def model_from_json(filepath: str) -> keras.Model:
    with open(filepath, "r") as f:
        config = json.load(f)
        return keras.models.model_from_json(config)