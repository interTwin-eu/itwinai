# from dataclasses import dataclass


class ServerOptions:
    host: str
    port: int

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port


class ClientOptions:
    url: str

    def __init__(self, url: str) -> None:
        self.url = url
