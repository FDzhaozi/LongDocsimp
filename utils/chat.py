import openai


from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils import config

openai.api_key = config.OPENAI_API_KEY
temperature = config.temperature
@retry(wait=wait_random_exponential(min=60, max=61), stop=stop_after_attempt(6))
def askChatGPT(messages):
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature)
    return response['choices'][0]['message']['content']



@retry(wait=wait_random_exponential(min=60, max=61), stop=stop_after_attempt(6))
def askLongChatGPT(messages):
    MODEL = "gpt-3.5-turbo-16k"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=temperature)
    return response['choices'][0]['message']['content']



def askDaVinci(prompt):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=1,
        max_tokens=1710,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    return response['choices'][0]['text']