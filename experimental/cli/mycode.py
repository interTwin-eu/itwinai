# from dataclasses import dataclass
from itwinai.components import BaseComponent


class ServerOptions(BaseComponent):
    host: str
    port: int

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port

    def execute():
        ...


class ClientOptions(BaseComponent):
    url: str

    def __init__(self, url: str) -> None:
        self.url = url

    def execute():
        ...


class ServerOptions2(BaseComponent):
    host: str
    port: int

    def __init__(self, client: ClientOptions) -> None:
        self.client = client

    def execute():
        ...
