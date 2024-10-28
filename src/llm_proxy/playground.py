from initialize_provider import LlmProxy, TandemProxy
from time import perf_counter


import asyncio

from time import perf_counter, sleep


def test_chat_shopping_list():

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


async def test_speed_async():

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


def test_vision():
    provider = "together"
    modality = "vision"
    model = ""
    llm_obj = LlmProxy(provider=provider, modality=modality, model=model)

    import os
    import subprocess

    image_name = "ldrts.png"

    image_path = os.path.join(os.path.expanduser("~"), "Pictures", "Screenshots", image_name)

    # root_path = eval(f"subprocess.getoutput('pwd')")
    # print(root_path)

    response = llm_obj.analyze_image(image_path)

    print(response)


def test_image_generation():

    from PIL import Image
    import glob
    import os

    provider = "together"
    modality = "image"
    model = ""
    llm_obj = LlmProxy(provider=provider, modality=modality, model=model)

    image_prompt = "imagine a cartoon scenario where an astronaut is holding a wounded horse in its arms."
    output_path = "./"
    output_name = "picture"
    image_out = llm_obj.get_image(image_prompt=image_prompt, output_path=output_path, output_name=output_name)

    list_of_files = glob.glob(output_path + output_name + "*")  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)

    # image_name = [filename for filename in os.listdir(output_path) if filename.startswith(output_name)]
    # print(image_name)
    img = Image.open(output_path + latest_file)
    img.show()

    # print(image_out)


def test_get_image_and_analyze_it_loop():
    from PIL import Image
    import glob
    import os
    import ast

    provider = "together"
    modality = "vision"
    model = ""
    llm_vision = LlmProxy(provider=provider, modality=modality, model=model)

    provider = "together"
    modality = "image"
    model = ""
    llm_image = LlmProxy(provider=provider, modality=modality, model=model)

    provider = "groq"
    modality = "chat"
    model = ""
    llm_chat = LlmProxy(provider=provider, modality=modality, model=model)

    provider = "together"
    modality = "chat"
    model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    llm_chat_405 = LlmProxy(provider=provider, modality=modality, model=model)

    original_image_prompt = "a horse riding on the back of an astronaut."
    output_path = "./noSubmit/images/"
    output_name = "horse_riding_astronaut"

    user_reflection_prompt = f"""
    The user wants to generate an image with the following ideas between three backticks:
    ```
    {original_image_prompt}
    ```
    Output a reflection about the key elements of the idea that should be conveyed in the image.
    
    Do not output anything else.
    Do not output verbose.
    """

    image_prompt = llm_chat_405.get_completion(user_prompt=user_reflection_prompt)

    achieved_goal = False
    while not achieved_goal:
        _ = llm_image.get_image(image_prompt=image_prompt, output_path=output_path, output_name=output_name)

        list_of_files = glob.glob(output_path + output_name + "*")  # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)

        img = Image.open(output_path + latest_file)
        img.show()

        user_image_prompt = f"""
        Follow the instructions below:
        - The target is to generate an image coherent with the following prompt between three backticks:
        ```
            {original_image_prompt}
        ```
        - Evaluate if the image is coherent with the target. 
        - Provide a description of the image.
        
        - The Image Prompts between the symbols <> were used to generate the image. 
        - Reflect what could be a better prompt to generate a more coherent image.
        - Suggest a better prompt to generate a more coherent image.
        
        Image Prompt
        <
        {image_prompt}
        >
        
        Format your response as a python dictionary with the following key and values:
        
            "evaluation": Evaluate if the image is coherent with the prompt. 
            "evaluation_decision": True or False
            "image_description": Provide a description of the image.
            "prompt_reflection": Reflect what could be a better prompt to generate a more coherent image.
            "prompt_suggestion": Suggest a better prompt to generate a more coherent image.
        
        """
        vision_response = llm_vision.analyze_image(user_image_prompt=user_image_prompt, image_path=latest_file)

        user_chat_prompt = f"""
        Follow the instructions below:
        - Format the Evaluation Text between the symbols <> into a json file. 
        
        Evaluation Text
        <
        {vision_response}
        >
        
        - Format your response as a python dictionary with the following keys and values:
        
            "evaluation": Concise evaluation about the image being coherent with the prompt. 
            "evaluation_decision": True or False
            "image_description": Concise description of the image.
            "prompt_reflection": Concise reflection about what could be a better prompt to generate a more coherent image.
            "prompt_suggestion": Suggestion for a better prompt to generate a more coherent image.
            
        - Do not use verbose.
        - Do not output anything else.
        - Only output a python dictionary.
        
        """

        system_prompt = "You are an expert formatting text into python dictionary."

        chat_response = llm_chat.get_completion(system_prompt=system_prompt, user_prompt=user_chat_prompt)
        dict_resp = ast.literal_eval(chat_response)

        print(chat_response)

        if not dict_resp["evaluation_decision"]:
            image_prompt = dict_resp["prompt_suggestion"]
        else:
            achieved_goal = True


def test_image_prompt_reflection():
    original_image_prompt = "a horse riding on the back of an astronaut."

    provider = "together"
    modality = "chat"
    model = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo"
    llm_chat_405 = LlmProxy(provider=provider, modality=modality, model=model)

    user_reflection_prompt = f"""
    The user wants to generate an image with the following ideas between three backticks:
    ```
    {original_image_prompt}
    ```
    Output a reflection about the key elements of the idea that should be conveyed in the image.

    """

    response_chat = llm_chat_405.get_completion(user_prompt=user_reflection_prompt)
    print(response_chat)


if __name__ == "__main__":
    # asyncio.run(test_speed_async())
    # test_vision()
    # test_image_generation()
    test_get_image_and_analyze_it_loop()
    # test_image_prompt_reflection()
    # test_chat_shopping_list()
