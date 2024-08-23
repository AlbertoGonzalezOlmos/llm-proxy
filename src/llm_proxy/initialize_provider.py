from groq import Groq
from together import Together

import asyncio
from dotenv import load_dotenv
import os
from typing import Literal, Union
from abc import ABC
import datetime
from time import perf_counter, sleep
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
        self.timer_start: float
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
                        self.model = "whisper-large-v3"
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
            f"Provider: '{self.provider}' was initialized for modality: '{self.modality}' with model '{self.model}'."
        )

    def logging_initialize(self) -> None:

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.log = logging.getLogger(__name__)

    def error_modality_not_supported(self) -> None:
        self.log.error(f"Modality: '{self.modality}' is not supported by '{self.provider}'.")

    def request_per_minute_limit_initialize(self) -> None:
        self.is_limit_reached = False

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
        self.request_per_minute_limit += 1
        self.log.info(
            f" - timeoff set to: {(60 / self.request_per_minute_limit):.2f} seconds.",
        )

    def set_timer(self) -> None:
        self.timer_start = perf_counter()

    def get_timer(self) -> int:
        return perf_counter() - self.timer_start

    def rpm_wait(self, queue_number: int = 0) -> None:
        time_left = 0
        if queue_number > 0:
            time_left = (60 / self.request_per_minute_limit) * queue_number
        else:
            time_left = (60 / self.request_per_minute_limit) - self.get_timer()
            if time_left < 0:
                time_left = 0
        sleep(time_left)
        self.is_limit_reached = False

    def time_off(self, queue_number: int = 0) -> None:
        if self.is_limit_reached:
            self.rpm_wait(queue_number)
        self.set_timer()
        self.is_limit_reached = True

    def tokenizer_initialize(self) -> None:
        self.session_start = llm_proxy_time_string()
        self.llm_messages_count_tokens = 0
        self.llm_response_count_tokens = 0

    async def count_async_tokens(self, llm_message: str, llm_response: str) -> None:
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

    def get_image(
        self, image_prompt: str, width: int = 1024, height: int = 1024, steps: int = 40, n: int = 4, seed: int = 6439
    ) -> str:

        response = self.client.images.generate(
            prompt=image_prompt,
            model=self.model,
            width=width,
            height=height,
            steps=steps,
            n=n,
            seed=seed,
        )

        image_out = response.data[0].b64_json

        im = Image.open(BytesIO(base64.b64decode(image_out)))
        name_output_file = f"{llm_proxy_time_string()}_{self.model}.jpg".replace("/", "_").replace("'", "")
        im.save(name_output_file, "JPEG")

        return image_out

    def audio_to_text(self, filepath):

        with open(filepath, "rb") as file:
            translation = self.client.audio.translations.create(
                file=(filepath, file.read()),
                model=self.model,
            )
        return translation.text

    async def get_async_completion_bulk(
        self,
        system_prompt_list: list[str] = [],
        user_prompt_list: list[str] = [],
        temperature_list: list[int] = [],
    ) -> tuple[str]:
        if not user_prompt_list:
            raise ValueError("User_prompt_list is empty.")
        if not system_prompt_list:
            system_prompt_list = [""] * len(user_prompt_list)
        if not temperature_list:
            temperature_list = [0] * len(user_prompt_list)
        self.set_timer()
        return await asyncio.gather(
            *[
                self.get_async_completion(
                    system_prompt=system_prompt_list[idx_item],
                    user_prompt=user_prompt,
                    temperature=temperature_list[idx_item],
                    queue_number=idx_item,
                )
                for idx_item, user_prompt in enumerate(user_prompt_list)
            ]
        )

    async def get_async_completion(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: int = 0,
        queue_number: int = 0,
    ) -> str:
        self.time_off(queue_number)
        # self.rpm_wait(queue_number)
        self.log.info(
            f" - elapsed time: {self.get_timer()}, should be >= than: {(60 / self.request_per_minute_limit):.2f}."
        )
        llm_response = await asyncio.to_thread(self.get_completion, system_prompt, user_prompt, temperature)
        # self.set_timer()
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

        self.time_off()

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

    def client_initialize(self) -> Union[Together, Groq]:
        load_dotenv()
        match self.model:
            case "llama-3.1-8b":
                self.groq_provider = LlmProxy(provider="groq", modality="chat", model="llama-3.1-8b-instant")
                self.together_provider = LlmProxy(
                    provider="together", modality="chat", model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                )
            case "llama-3.1-70b":
                self.groq_provider = LlmProxy(provider="groq", modality="chat", model="llama-3.1-70b-versatile")
                self.together_provider = LlmProxy(
                    provider="together", modality="chat", model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
                )
        self.list_providers = [self.groq_provider, self.together_provider]
        self.number_of_providers = len(self.list_providers)
        self.current_provider = 0
        self.log.info(f"LLM Tandem initialized.")

    def logging_initialize(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self.log = logging.getLogger(__name__)

    async def get_async_completion_bulk(
        self,
        system_prompt_list: list[str] = [],
        user_prompt_list: list[str] = [],
        temperature_list: list[int] = [],
    ) -> tuple[str]:
        if not user_prompt_list:
            raise ValueError("User_prompt_list is empty.")
        if not system_prompt_list:
            system_prompt_list = [""] * len(user_prompt_list)
        if not temperature_list:
            temperature_list = [0] * len(user_prompt_list)
        return await asyncio.gather(
            *[
                self.get_async_completion(
                    system_prompt=system_prompt_list[idx_item],
                    user_prompt=user_prompt,
                    temperature=temperature_list[idx_item],
                )
                for idx_item, user_prompt in enumerate(user_prompt_list)
            ]
        )

    async def get_async_completion(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: int = 0,
    ) -> str:

        return await asyncio.to_thread(self.get_completion, system_prompt, user_prompt, temperature)

    def get_completion(
        self,
        system_prompt: str = "",
        user_prompt: str = "",
        temperature: int = 0,
    ) -> str:
        is_request_successful = False
        attempts = 1
        while not is_request_successful:
            try:
                self.log.info(f"Using: '{self.list_providers[self.current_provider].provider}' provider...")
                llm_response = self.list_providers[self.current_provider].get_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
                is_request_successful = True
            except:
                self.log.warning(
                    f"Request failed with: '{self.list_providers[self.current_provider].provider}' provider. Attempt: {attempts}."
                )
                self.list_providers[self.current_provider].rpm_wait()
                self.log.warning(
                    f"Request failed with: '{self.list_providers[self.current_provider].provider}' provider."
                )
                attempts += 1
        self.next_provider()

        return llm_response

    def next_provider(self) -> None:
        self.current_provider += 1
        if self.current_provider >= self.number_of_providers:
            self.current_provider = 0


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

    from prompts_to_test import list_of_list_of_requests, list_of_requests_tandem

    results_list_dict = []

    # provider = "together"
    # modality = "chat"
    # model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
    # llmObj = LlmProxy(provider=provider, modality=modality, model=model)

    # time_start = perf_counter()
    # _ = await asyncio.gather(
    #     *[llmObj.get_async_completion_bulk(user_prompt_list=request_list) for request_list in list_of_list_of_requests]
    # )
    # strategy = f"bulk, provider: {provider}"
    # elapsed_time_int = perf_counter() - time_start
    # elapsed_time = f"{elapsed_time_int:.2f}"
    # time_per_request_float = elapsed_time_int / len(list_of_requests_tandem)
    # time_per_request = f"{time_per_request_float:.2f}"
    # results_list_dict.append(
    #     {
    #         "strategy": strategy,
    #         "elapsed_time": elapsed_time,
    #         "number_of_requests": len(list_of_requests_tandem),
    #         "time_per_request": time_per_request,
    #     }
    # )

    # provider = "groq"
    # modality = "chat"
    # model = "llama-3.1-8b-instant"
    # llmObj = LlmProxy(provider=provider, modality=modality, model=model)

    # time_start = perf_counter()
    # _ = await asyncio.gather(
    #     *[llmObj.get_async_completion_bulk(user_prompt_list=request_list) for request_list in list_of_list_of_requests]
    # )
    # strategy = f"bulk, provider: {provider}"
    # elapsed_time_int = perf_counter() - time_start
    # elapsed_time = f"{elapsed_time_int:.2f}"
    # time_per_request_float = elapsed_time_int / len(list_of_requests_tandem)
    # time_per_request = f"{time_per_request_float:.2f}"
    # results_list_dict.append(
    #     {
    #         "strategy": strategy,
    #         "elapsed_time": elapsed_time,
    #         "number_of_requests": len(list_of_requests_tandem),
    #         "time_per_request": time_per_request,
    #     }
    # )

    llmObj_tandem = TandemProxy(model="llama-3.1-8b")
    list_results_tandem = []
    time_start = perf_counter()
    for i_list_item in list_of_requests_tandem:
        list_results_tandem.append(llmObj_tandem.get_completion(user_prompt=i_list_item))
    strategy = f"Tandem serial"
    elapsed_time_int = perf_counter() - time_start
    elapsed_time = f"{elapsed_time_int:.2f}"
    time_per_request_float = elapsed_time_int / len(list_of_requests_tandem)
    time_per_request = f"{time_per_request_float:.2f}"
    results_list_dict.append(
        {
            "strategy": strategy,
            "elapsed_time": elapsed_time,
            "number_of_requests": len(list_of_requests_tandem),
            "time_per_request": time_per_request,
        }
    )

    llmObj_tandem_2 = TandemProxy(model="llama-3.1-8b")
    list_results_tandem_2 = []
    time_start = perf_counter()
    for i_list_item in list_of_requests_tandem:
        list_results_tandem_2.append(await llmObj_tandem_2.get_async_completion(user_prompt=i_list_item))
    strategy = f"Tandem async"
    elapsed_time_int = perf_counter() - time_start
    elapsed_time = f"{elapsed_time_int:.2f}"
    time_per_request_float = elapsed_time_int / len(list_of_requests_tandem)
    time_per_request = f"{time_per_request_float:.2f}"
    results_list_dict.append(
        {
            "strategy": strategy,
            "elapsed_time": elapsed_time,
            "number_of_requests": len(list_of_requests_tandem),
            "time_per_request": time_per_request,
        }
    )

    llmObj_tandem_3 = TandemProxy(model="llama-3.1-8b")
    time_start = perf_counter()
    _ = await asyncio.gather(
        *[
            llmObj_tandem_3.get_async_completion_bulk(user_prompt_list=request_list)
            for request_list in list_of_list_of_requests
        ]
    )
    strategy = f"Tandem async bulk"
    elapsed_time_int = perf_counter() - time_start
    elapsed_time = f"{elapsed_time_int:.2f}"
    time_per_request_float = elapsed_time_int / len(list_of_requests_tandem)
    time_per_request = f"{time_per_request_float:.2f}"
    results_list_dict.append(
        {
            "strategy": strategy,
            "elapsed_time": elapsed_time,
            "number_of_requests": len(list_of_requests_tandem),
            "time_per_request": time_per_request,
        }
    )

    print("####################################### \n #########################################")
    for dict_entry in results_list_dict:
        print(
            "Proxy strategy: {} \n".format(dict_entry["strategy"]),
            " - Number of requests: {} \n".format(dict_entry["number_of_requests"]),
            " - Elapsed time: {} seconds \n".format(dict_entry["elapsed_time"]),
            " - Time per request: {} seconds \n".format(dict_entry["time_per_request"]),
        )


if __name__ == "__main__":
    asyncio.run(main_async())

    # main()
