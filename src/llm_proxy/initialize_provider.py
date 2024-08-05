from groq import Groq
from together import Together

import asyncio
from dotenv import load_dotenv
import os
from typing import Literal, Union
from abc import ABC
import datetime
from time import perf_counter
import logging

from io import BytesIO
import base64
from PIL import Image

list_providers = Literal["groq", "together"]
list_modalities = Literal["chat", "image", "stt", "code"]


class LlmProxy(ABC):
    def __init__(self, provider: list_providers, modality: list_modalities, model: str = ""):
        self.logging_initialize()
        self.provider = provider
        self.modality = modality
        self.model = model
        self.client_initialize()
        self.request_per_minute_limit_initialize()
        match modality:
            case "chat":
                self.tokenizer_initialize()

    def client_initialize(self) -> Union[Together, Groq]:
        load_dotenv()

        match self.provider:
            case "groq":
                api_key = os.environ.get("GROQ_API_KEY")
                self.client = Groq(api_key=api_key)
                match self.modality:
                    case "chat":
                        if self.model == "":
                            self.model = "llama-3.1-70b-versatile"
                        # "llama-3.1-405b-reasoning" -> not working?
                        # "llama-3.1-70b-versatile"
                        # "llama-3.1-8b-instant"
                        # "llama3-70b-8192"
                        # "mixtral-8x7b-32768"
                        # "gemma-7b-it"
                    case "image":
                        return self.error_modality_not_supported()
                    case "stt":
                        return self.error_modality_not_supported()
                        # self.model = "whisper-large-v3"
                    case "code":
                        return self.error_modality_not_supported()

            case "together":
                api_key = os.environ.get("TOGETHER_API_KEY")
                self.client = Together(api_key=api_key)
                match self.modality:
                    case "chat":
                        if self.model == "":
                            self.model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                        # "mistralai/Mixtral-8x22B-Instruct-v0.1"
                        # "Qwen/Qwen1.5-110B-Chat"
                    case "image":
                        if self.model == "":
                            self.model = "stabilityai/stable-diffusion-xl-base-1.0"
                            # "prompthero/openjourney"
                            # "runwayml/stable-diffusion-v1-5"
                            # "SG161222/Realistic_Vision_V3.0_VAE"
                            # "stabilityai/stable-diffusion-2-1"
                            # "stabilityai/stable-diffusion-xl-base-1.0"
                            # "wavymulder/Analog-Diffusion"
                    case "stt":
                        return self.error_modality_not_supported()
                    case "code":
                        return self.error_modality_not_supported()
                        # "Qwen/Qwen1.5-72B"
                        # "mistralai/Mistral-7B-v0.1"
                        # "mistralai/Mixtral-8x7B-v0.1"

        self.log.info(
            f"Provider: '{self.provider}' was initialized for modality: '{self.modality}' with model '{self.model}'..."
        )

    def logging_initialize(self) -> None:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.log = logging.getLogger(__name__)
        self.log.info("Starting LLM Proxy...")

    def error_modality_not_supported(self) -> None:
        self.log.error(f"Modality: '{self.modality}' is not supported by '{self.provider}'.")

    def request_per_minute_limit_initialize(self) -> None:
        match self.provider:
            case "groq":
                match self.model:
                    case "llama-3.1-70b-versatile":
                        self.request_per_minute_limit = 100
                    case "whisper-large-v3":
                        self.request_per_minute_limit = 20
                    case _:
                        self.request_per_minute_limit = 30
            case "together":
                self.request_per_minute_limit = 60

    async def rpm_wait(self) -> bool:
        print(f"will await {60 / self.request_per_minute_limit} seconds ...")
        time_before = perf_counter()
        await asyncio.sleep(60 / self.request_per_minute_limit)
        print(f"... waited: {perf_counter() - time_before} seconds.")
        return True

    def tokenizer_initialize(self) -> None:
        self.session_start = llm_proxy_time_string()
        self.llm_messages_count_tokens = 0
        self.llm_response_count_tokens = 0

    async def count_async_tokens(self, llm_message: str, llm_response: str) -> None:
        print("count!")
        await asyncio.to_thread(self.count_tokens, llm_message, llm_response)

    def count_tokens(self, llm_message: str, llm_response: str) -> None:
        self.llm_messages_count_tokens += len(llm_message.split())
        self.llm_response_count_tokens += len(llm_response.split())

    def get_token_count(self) -> list[int, int]:
        token_count_dict = {}
        token_count_dict = {
            "session_start": self.session_start,
            "llm_messages_count_tokens": self.llm_messages_count_tokens,
            "llm_response_count_tokens": self.llm_response_count_tokens,
        }
        return token_count_dict

    def get_image(self, image_prompt: str):

        response = self.client.images.generate(
            prompt=image_prompt,
            model=self.model,
            width=1024,
            height=1024,
            steps=40,
            n=4,
            seed=6439,
        )

        image_out = response.data[0].b64_json

        im = Image.open(BytesIO(base64.b64decode(image_out)))
        im.save("picture_out.jpg", "JPEG")

        return image_out

    async def get_async_completion(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: int = 0,
    ) -> str:

        if await self.rpm_wait():
            llm_response = await asyncio.to_thread(self.get_completion, system_prompt, user_prompt, temperature)
            await self.count_async_tokens(system_prompt + user_prompt, llm_response)
        return llm_response

    def get_completion(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: int = 0,
    ) -> str:

        if not system_prompt:
            system_prompt = "You are a useful assistant."
        llm_message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        llm_response = ""

        match self.provider:
            case "groq" | "together" | "openai":

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=llm_message,
                    temperature=temperature,
                )

                llm_response = completion.choices[0].message.content

            case "anthropic":

                completion = self.client.messages.create(
                    max_tokens=1000,
                    model=self.model,
                    temperature=temperature,
                    system=llm_message[0]["content"],
                    messages=[
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": user_prompt}],
                        }
                    ],
                )

                llm_response = completion.content[0].text

        return llm_response


def llm_proxy_time_string() -> str:
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.timestamp()
    formatted_string = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d_%Hh%Mm%Ss")
    return formatted_string


list_tandem_models = Literal["llama-3.1-70b", "llama-3.1-8b"]


class TandemProxy(ABC):
    def __init__(self, model: list_tandem_models):
        self.logging_initialize()
        self.model = model
        self.client_initialize()
        self.request_per_minute_limit_initialize()
        self.tokenizer_initialize()

    def client_initialize(self) -> Union[Together, Groq]:
        load_dotenv()


def main():

    shopping_list = f"""

    4 apples, 
    500g chicken,
    2 apples,
    100 chicken,
    1 kg carrots,
    200 g walnuts,

    """
    provider = "together"
    modality = "chat"
    model = ""
    llmObj = LlmProxy(provider=provider, modality=modality, model=model)
    print("start")
    completion = llmObj.get_completion(user_prompt=shopping_list)
    print("finish")
    print(completion)


async def main_async():
    shopping_list = f"""

    4 apples, 
    500g chicken,
    2 apples,
    100 chicken,
    1 kg carrots,
    200 g walnuts,

    """
    provider = "together"
    modality = "chat"
    model = ""
    llmObj = LlmProxy(provider=provider, modality=modality, model=model)
    print("start")
    completion = await llmObj.get_async_completion(user_prompt=shopping_list)
    print("finish")
    print(completion)
    print("finish finish")


if __name__ == "__main__":
    # asyncio.run(main_async())

    main()
