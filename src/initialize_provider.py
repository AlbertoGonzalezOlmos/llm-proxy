from groq import Groq
from together import Together

# from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import Literal, Union
from abc import ABC
import datetime

from file_read_write import read_file, write_file

from file_paths import create_output_path

list_providers = Literal["groq", "together"]


class LlmProxy(ABC):
    def __init__(self, provider: list_providers, model: str = "") -> None:
        self.provider = provider
        self.model = model
        self.client_initialize(self.provider, self.model)
        self.tokenizer_initialize()

    def client_initialize(
        self, provider: list_providers, model: str
    ) -> Union[Together, Groq]:
        load_dotenv()

        if provider == "groq":
            api_key = os.environ.get("GROQ_API_KEY")
            self.client = Groq(api_key=api_key)
            if model == "":
                self.model = "llama-3.1-70b-versatile"
            # "llama-3.1-70b-versatile"
            # "llama-3.1-8b-instant"
            # "llama3-70b-8192"
            # "mixtral-8x7b-32768"
            # "gemma-7b-it"

        elif provider == "together":
            api_key = os.environ.get("TOGETHER_API_KEY")
            self.client = Together(api_key=api_key)
            if model == "":
                self.model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
            # "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
            # "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"

        # elif provider == "openai":
        #     api_key = os.environ.get("OPENAI_API_KEY")
        #     self.client = OpenAI(api_key=api_key)
        #     self.model = "gpt-4-1106-preview"
        print(f"Provider: '{provider}' was initialized with model '{self.model}'...")

    def tokenizer_initialize(self) -> None:
        self.session_start = llm_proxy_time_string()
        self.llm_messages_count_tokens = 0
        self.llm_response_count_tokens = 0
        pass

    def count_tokens(self, llm_message: str = "", llm_response: str = "") -> None:
        if llm_message and llm_response:
            self.llm_messages_count_tokens += len(llm_message.split())
            self.llm_response_count_tokens += len(llm_response.split())
        pass

    def get_token_count(self) -> list[int, int]:
        token_count_dict = {}
        token_count_dict = {
            "session_start": self.session_start,
            "llm_messages_count_tokens": self.llm_messages_count_tokens,
            "llm_response_count_tokens": self.llm_response_count_tokens,
        }
        return token_count_dict

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

        if self.provider in ["groq", "together", "openai"]:

            completion = self.client.chat.completions.create(
                model=self.model,
                messages=llm_message,
                temperature=temperature,
            )

            llm_response = completion.choices[0].message.content

        if self.provider == "anthropic":

            completion = self.client.messages.create(
                max_tokens=1000,
                model=self.model,
                temperature=temperature,
                system=llm_message[0]["content"],
                messages=[
                    {"role": "user", "content": [{"type": "text", "text": user_prompt}]}
                ],
            )

            llm_response = completion.content[0].text

        llm_message_result_list = [
            ", ".join([f"{key}: {value}" for key, value in dictionary.items()])
            for dictionary in llm_message
        ]
        llm_message_result_string = ", ".join(llm_message_result_list)
        self.count_tokens(llm_message_result_string, llm_response)

        return llm_response

    def prompt_chain(
        self,
        prompt: str,
        output_response_name: str,
        input_context_path_name: str = "",
    ) -> str:
        input_context_text = ""
        if input_context_path_name:
            input_context_text = read_file(input_context_path_name)

        output_response_path = create_output_path(output_response_name)
        output_response_path_name = output_response_path + output_response_name

        print(f"output path: {output_response_path_name}")

        prompt = f"{prompt}<{input_context_text}>"

        response = self.get_completion(prompt=prompt)

        write_file(output_response_path_name, response)

        return response


def llm_proxy_time_string() -> str:
    current_datetime = datetime.datetime.now()
    timestamp = current_datetime.timestamp()
    formatted_string = datetime.datetime.fromtimestamp(timestamp).strftime(
        "%Y-%m-%d_%Hh%Mm%Ss"
    )
    return formatted_string


def main():
    shopping_list = f"""

    4 apples, 
    500g chicken,
    2 apples,
    100 chicken,
    1 kg carrots,
    200 g walnuts,

    """

    llmObj = LlmProxy("together")
    completion = llmObj.get_completion(user_prompt=shopping_list)

    print(completion)


if __name__ == "__main__":
    main()
