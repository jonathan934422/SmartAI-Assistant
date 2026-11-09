from groq import Groq

from config.settings import GROQ_API_KEY


client = Groq(
    api_key=GROQ_API_KEY
)


MODEL_NAME = "openai/gpt-oss-120b"


def generate_response(messages):
    """
    Generate a normal AI response using Groq.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_tokens=2048,
        stream=True,
    )

    for chunk in response:

        content = chunk.choices[0].delta.content

        if content:
            yield content