import os
import yaml
from typing import Any, Union
from argparse import Namespace
from abc import ABC, abstractmethod

import openai

class Model(ABC):
    """A model for generating a response to a given prompt."""

    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: Any) -> str:
        """Generates a response to a given prompt."""
        raise NotImplementedError
    
class OpenAIModel(Model):
    """A model that uses an API (e.g., OpenAI APIs) to generate a response to a given prompt."""
    chatcompletion_models = {"gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613"}
    completion_models = {"text-davinci-003", "text-curie-001"}

    def __init__(
        self,
        config: Union[dict, Namespace],
    ):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if isinstance(config, Namespace):
            config = vars(config)
        self.config = config
    
    def generate(self, prompt: Any) -> str:
        """Generates a response to a given prompt."""
        if self.config["model"] in self.chatcompletion_models:
            return self.chatcompletion(prompt)
        elif self.config["model"] in self.completion_models:
            return self.completion(prompt)
        else:
            raise ValueError(f"Invalid model: {self.config['model']}")
    
    def chatcompletion(self, prompt: list[dict[str, str]]) -> str:
        """POST to the https://api.openai.com/v1/chat/completions endpoint."""
        completion = openai.ChatCompletion.create(
            messages=prompt,
            **self.config
        )
        return completion.choices[0].message.content

    def completion(self, prompt: str) -> str:
        """POST to the https://api.openai.com/v1/completions endpoint."""
        completion = openai.Completion.create(
            prompt=prompt,
            **self.config
        )
        return completion.choices[0].text

class LocalModel(Model):
    """A model that uses a local model (e.g., LLaMA) to generate a response to a given prompt."""

    def __init__(self):
        raise NotImplementedError

# manual testing
if __name__ == "__main__":
    with open("../../experiments/configs/debug.yml") as f:
        args = yaml.safe_load(f)
    
    model = OpenAIModel(args["patient"]["model_config"])
    print(model.generate(prompt=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Hello!"
        }
    ]))
