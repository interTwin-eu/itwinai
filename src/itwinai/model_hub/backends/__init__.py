from .itwinai_hub import AIModelHubBackend


def get_backend(name: str, config: dict):
    if name == "ai-model-hub":
        return AIModelHubBackend(config)
    else:
        raise ValueError(f"Unknown backend: {name}")
