from abc import ABC, abstractmethod

class Model(ABC):
    """A model for generating a response to a given prompt."""

    def __init__(self):
        raise NotImplementedError
    
class APIModel(Model):
    """A model that uses an API (e.g., OpenAI APIs) to generate a response to a given prompt."""

    def __init__(self):
        pass

class LocalModel(Model):
    """A model that uses a local model (e.g., LLaMA) to generate a response to a given prompt."""

    def __init__(self):
        raise NotImplementedError
