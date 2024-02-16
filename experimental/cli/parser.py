"""
Example of dynamic override of config files with (sub)class arguments,
and variable interpolation with omegaconf.

Run with:
>>> python parser.py

Or (after clearing the arguments in parse_args(...)):
>>> python parser.py --config example.yaml --server.port 212
See the help page of each class:
>>> python parser.py --server.help mycode.ServerOptions
"""

from jsonargparse import ArgumentParser, ActionConfigFile
from mycode import ServerOptions, ClientOptions

if __name__ == "__main__":
    parser = ArgumentParser(parser_mode="omegaconf")
    parser.add_subclass_arguments(ServerOptions, "server")
    parser.add_subclass_arguments(ClientOptions, "client")
    parser.add_argument("--config", action=ActionConfigFile)

    # Example of dynamic CLI override
    # cfg = parser.parse_args(["--config=example.yaml", "--server.port=212"])
    cfg = parser.parse_args()
    cfg = parser.instantiate_classes(cfg)
    print(cfg.client)
    print(cfg.client.url)
    print(cfg.server.port)
