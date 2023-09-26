import yaml


# Parse (part of) YAML loaded in memory
def parse_pipe_config(yaml_file, parser):
    with open(yaml_file, "r", encoding="utf-8") as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            print(exc)
            raise exc

    return parser.parse_object(config)
