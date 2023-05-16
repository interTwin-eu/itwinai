from jsonargparse import ArgumentParser
import yaml

from itwinai.plmodels.base import ItwinaiBasePlModel

# Ref:
# https://jsonargparse.readthedocs.io/en/stable/#class-type-and-sub-classes

parser = ArgumentParser()

parser.add_argument('--car', type=str)
parser.add_argument('--number', type=int)
parser.add_subclass_arguments(ItwinaiBasePlModel, 'model')


# Parse (part of) YAML loaded in memory
with open('config.yml', "r", encoding="utf-8") as yaml_file:
    try:
        train_config = yaml.safe_load(yaml_file)
    except yaml.YAMLError as exc:
        print(exc)
        raise exc
cfg = parser.parse_object(train_config)

# # Parse whole YAML from file
# cfg = parser.parse_path('config.yml')

print(cfg.model.as_dict())

cfg = parser.instantiate_classes(cfg)
print(cfg.model)
