from groq import Groq

from config.settings import GROQ_API_KEY


client = Groq(
    api_key=GROQ_API_KEY
)


# Use the smaller model for web search.
# It supports Groq's built-in browser search.
MODEL_NAME = "openai/gpt-oss-20b"


def web_search_response(user_question):
    """
    Search the web and generate an answer
    for the user's latest question.
    """

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SmartAI Assistant with web search. "
                    "Use the browser search tool to find current "
                    "and accurate information from the web. "
                    "Answer the user's question clearly and simply. "
                    "Mention important sources or citations when available."
                ),
            },
            {
                "role": "user",
                "content": user_question,
            },
        ],
        tools=[
            {
                "type": "browser_search"
            }
        ],
        tool_choice="required",
        temperature=0.7,
        max_completion_tokens=1024,
        reasoning_effort="low",
        stream=False,
    )

    message = response.choices[0].message

    return message.content or ""