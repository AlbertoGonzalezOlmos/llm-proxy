from groq import Groq
from together import Together

import asyncio
from dotenv import load_dotenv
import os
from typing import Literal, Union
import datetime
from time import perf_counter, sleep
import logging

from io import BytesIO
import base64
from PIL import Image

list_providers = Literal["groq", "together"]
list_modalities = Literal["chat", "image", "stt", "code", "vision"]


class LlmProxy:
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
                    case "vision":
                        return self.error_modality_not_supported()

            case "together":
                api_key = os.environ.get("TOGETHER_API_KEY")
                self.client = Together(api_key=api_key)
                match self.modality:
                    case "chat":
                        if self.model == "":
                            self.model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
                        # "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
                        # "mistralai/Mixtral-8x22B-Instruct-v0.1"
                        # "Qwen/Qwen1.5-110B-Chat"
                    case "image":
                        if self.model == "":
                            self.model = "black-forest-labs/FLUX.1.1-pro"
                            # "black-forest-labs/FLUX.1.1-pro"
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
                    case "vision":
                        if self.model == "":
                            self.model = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
                        # "meta-llama/Llama-Vision-Free"
                        # "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
                        # "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo"

        self.log.info(
            f"Provider: '{self.provider}' was initialized for modality: '{self.modality}' with model '{self.model}'."
        )

    def logging_initialize(self) -> None:

        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
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
        self,
        image_prompt: str,
        output_path: str = "./",
        output_name: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 1,
        n: int = 1,
        seed: int = 6439,
    ) -> str:

        response = self.client.images.generate(
            prompt=image_prompt,
            model=self.model,
            width=width,
            height=height,
            steps=steps,
            n=n,
            seed=seed,
            response_format="b64_json",
        )

        image_out = response.data[0].b64_json

        im = Image.open(BytesIO(base64.b64decode(image_out)))
        if not output_path.endswith("/"):
            output_path += "/"
        name_output_file = f"{output_name+llm_proxy_time_string()}_{self.model}.jpg".replace("/", "_").replace("'", "")
        im.save(output_path + name_output_file, "JPEG")

        return image_out

    async def audio_to_text_async(self, filepath: str) -> str:
        return await asyncio.to_thread(self.audio_to_text, filepath)

    def audio_to_text(self, filepath: str) -> str:

        self.time_off()

        with open(filepath, "rb") as file:
            translation = self.client.audio.translations.create(
                file=(filepath, file.read()),
                model=self.model,
            )
        return translation.text

    def analyze_image(self, image_path: str = "", system_image_prompt: str = "", user_image_prompt: str = "") -> str:

        if not image_path:
            url_text = "https://napkinsdev.s3.us-east-1.amazonaws.com/next-s3-uploads/d96a3145-472d-423a-8b79-bca3ad7978dd/trello-board.png"

        else:
            with open(image_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")
                url_text = f"data:image/png;base64,{base64_image}"

        if not system_image_prompt:
            system_image_prompt = (
                "You are an image analyst.  Your goal is to describe what is in the image provided as a file."
            )
        if not user_image_prompt:
            user_image_prompt = "What is in this image?"

        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_image_prompt,
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_image_prompt},
                    {"type": "image_url", "image_url": {"url": url_text}},
                ],
            },
        ]
        # {
        #     "role": "system",
        #     "content": [
        #         {
        #             "type": "text",
        #             "text": system_image_prompt,
        #         }
        #     ],
        # },
        # {
        #     "role": "user",
        #     "content": [
        #         {"type": "text", "text": user_image_prompt},
        #         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        #     ],
        # }

        # {
        #         "role": "assistant",
        #         "content": "The image appears to be a satellite view of a desert landscape with a body of water. The left side of the image shows a vast expanse of sandy terrain, dotted with small green patches that could be vegetation or trees. A long, straight line runs diagonally across this section, possibly indicating a road or a boundary.\n\nOn the right side of the image, there is a large body of water, which could be a lake, river, or ocean. The water‛s edge is irregularly shaped, with several inlets and peninsulas visible. The surrounding terrain is also sandy, but it appears more rugged and rocky than the area on the left side of the image.\n\nOverall, the image suggests a harsh, arid environment with limited vegetation and a significant body of water. The exact location and purpose of the image are unclear without additional context."
        # }

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=512,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            stop=["<|eot_id|>", "<|eom_id|>"],
            stream=False,
        )
        return response.choices[0].message.content

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
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        #     datefmt="%Y-%m-%d %H:%M:%S",
        # )
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

                llm_response = self.list_providers[self.current_provider].get_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature,
                )
                self.log.info(f"'{self.list_providers[self.current_provider].provider}' request -> OK")
                is_request_successful = True
            except:
                self.log.warning(
                    f"Request failed with: '{self.list_providers[self.current_provider].provider}'. Attempt: {attempts}."
                )
                self.list_providers[self.current_provider].rpm_wait()
                attempts += 1
        self.next_provider()

        return llm_response

    def next_provider(self) -> None:
        self.current_provider += 1
        if self.current_provider >= self.number_of_providers:
            self.current_provider = 0
