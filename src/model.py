import os
import ssl
import json
import time
import yaml
import urllib.request
from typing import Any, Union
from pathlib import Path
from argparse import Namespace
from abc import ABC, abstractmethod
from colorama import Fore, Style

import openai
import google.generativeai as palm
import google.api_core.exceptions

def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    max_retries: int = 10,
    errors: tuple = (
        openai.RateLimitError, openai.APIError,
        google.api_core.exceptions.ResourceExhausted, google.api_core.exceptions.ServiceUnavailable, google.api_core.exceptions.GoogleAPIError,
        urllib.error.HTTPError, urllib.error.URLError
    ),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        Fore.RED + f"Maximum number of retries ({max_retries}) exceeded." + Style.RESET_ALL
                    )
                # Increment the delay
                delay *= exponential_base
                # Sleep for the delay
                print(Fore.YELLOW + f"Error encountered. Retry ({num_retries}) after {delay} seconds..." + Style.RESET_ALL)
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e

    return wrapper

class Model(ABC):
    """A model for generating a response to a given prompt."""

    def __init__(self):
        raise NotImplementedError
    
    @abstractmethod
    def generate(self, prompt: Any) -> str:
        """Generates a response to a given prompt."""
        raise NotImplementedError

class PaLM2Model(Model):
    """A model that uses PaLM2 to generate a response to a given prompt."""
    chatcompletion_models = set()
    completion_models = {"models/text-bison-001"}

    def __init__(
        self,
        config: Union[dict, Namespace],
    ):
        palm.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        if isinstance(config, Namespace):
            config = vars(config)
        self.config = config

    @retry_with_exponential_backoff
    def generate(self, prompt: str) -> str:
        """Generate a response to a given prompt."""
        response = palm.generate_text(
            **self.config,
            prompt=prompt,
            safety_settings=[  # set all safety settings to 4 (no restrictions)
                {"category": "HARM_CATEGORY_DEROGATORY", "threshold": 4},
                {"category": "HARM_CATEGORY_TOXICITY", "threshold": 4},
                {"category": "HARM_CATEGORY_VIOLENCE", "threshold": 4},
                {"category": "HARM_CATEGORY_SEXUAL", "threshold": 4},
                {"category": "HARM_CATEGORY_MEDICAL", "threshold": 4},
                {"category": "HARM_CATEGORY_DANGEROUS", "threshold": 4}
            ]
        )
        return response.result

class OpenAIModel(Model):
    """A model that uses an API (e.g., OpenAI APIs) to generate a response to a given prompt."""
    chatcompletion_models = {"gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k", "gpt-3.5-turbo-16k-0613", "gpt-4", "gpt-4-0613", "gpt-4-32k", "gpt-4-32k-0613"}
    completion_models = {"gpt-3.5-turbo-instruct"}

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

    @retry_with_exponential_backoff
    def chatcompletion(self, prompt: list[dict[str, str]]) -> str:
        """POST to the https://api.openai.com/v1/chat/completions endpoint."""
        completion = openai.ChatCompletion.create(
            messages=prompt,
            **self.config
        )
        return completion.choices[0].message.content

    @retry_with_exponential_backoff
    def completion(self, prompt: str) -> str:
        """POST to the https://api.openai.com/v1/completions endpoint."""
        completion = openai.Completion.create(
            prompt=prompt,
            **self.config
        )
        return completion.choices[0].text

class LlamaModel(Model):
    completion_models = {"Llama-2-7b", "Llama-2-13b", "Llama-2-70b"}
    chatcompletion_models = {"Llama-2-7b-chat", "Llama-2-13b-chat", "Llama-2-70b-chat"}
    api_urls = {
        "Llama-2-7b": "https://Llama-2-7b-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/completions",
        "Llama-2-13b": "https://Llama-2-13b-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/completions",
        "Llama-2-70b": "https://Llama-2-70b-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/completions",
        "Llama-2-7b-chat": "https://Llama-2-7b-chat-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/chat/completions",
        "Llama-2-13b-chat": "https://Llama-2-13b-chat-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/chat/completions",
        "Llama-2-70b-chat": "https://Llama-2-70b-chat-sbhdemo-serverless.eastus2.inference.ai.azure.com/v1/chat/completions"
    }
    errors = (urllib.error.HTTPError, urllib.error.URLError)

    def __init__(
        self,
        config: Union[dict, Namespace]
    ):
        if isinstance(config, Namespace):
            config = vars(config)
        self.model = config["model"]
        self.config = config
        self.api_url = self.api_urls[self.model]
        self.api_key = Path(f"../api_keys/azure_{self.model}.txt").read_text().strip()
        if not self.api_key:
            raise Exception("A key should be provided to invoke the endpoint")
        self.headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ self.api_key)}
        self.allowSelfSignedHttps(True) # this line is needed if you use self-signed certificate in your scoring service.

    @retry_with_exponential_backoff
    def generate(self, prompt: Union[str, list[dict[str, str]]]) -> str:
        if self.model in self.completion_models:
            if type(prompt) != str:
                raise TypeError("Prompt must be a string for completion models.")
            return self.base_completion(prompt)["choices"][0]["text"]
        elif self.model in self.chatcompletion_models:
            if type(prompt) != list:
                raise TypeError("Prompt must be a list of messages for chat models.")
            res = self.chat_completion(prompt)
            assert res["choices"][0]["message"]["role"] == "assistant"
            return res["choices"][0]["message"]["content"]

    def base_completion(self, prompt: str) -> dict:
        req_data = self.config.copy()
        req_data["prompt"] = prompt
        req_data.pop("model")
        body = str.encode(json.dumps(req_data))
        req = urllib.request.Request(self.api_url, body, self.headers)
        try:
            response = urllib.request.urlopen(req)
            result = response.read()
            return json.loads(result.decode("utf-8"))
        except self.errors as error:
            print("The request failed with status code: " + str(error.code))
            raise error

    def chat_completion(self, messages: list[dict[str, str]]) -> dict:
        req_data = self.config.copy()
        req_data["messages"] = messages
        req_data.pop("model")
        body = str.encode(json.dumps(req_data))
        req = urllib.request.Request(self.api_url, body, self.headers)
        try:
            response = urllib.request.urlopen(req)
            result = response.read()
            return json.loads(result.decode("utf-8"))
        except self.errors as error:
            print("The request failed with status code: " + str(error.code))
            raise error

    @staticmethod
    def allowSelfSignedHttps(allowed):
        # bypass the server certificate verification on client side
        if allowed and not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None):
            ssl._create_default_https_context = ssl._create_unverified_context

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
