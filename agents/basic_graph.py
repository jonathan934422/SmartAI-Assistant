
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from core.groq_client import client, MODEL_NAME
from search.web_search import web_search_response
from rag.retriever import find_relevant_chunks
from rag.embeddings import create_embeddings


class AssistantState(TypedDict):
    user_message: str
    route: str
    response: str
    pdf_chunks: list[str]
    pdf_embeddings: Optional[object]


def router_node(state: AssistantState):
    message = state["user_message"].lower()

    rag_keywords = [
        "uploaded pdf", "my pdf", "this pdf",
        "uploaded document", "my document",
        "according to the pdf", "according to my pdf",
        "in the document", "from the pdf",
    ]

    web_keywords = [
        "today", "today's", "latest", "current",
        "recent", "live", "news", "weather",
        "stock price", "cricket score", "match score",
        "search the web", "search online",
    ]

    if any(word in message for word in rag_keywords):
        route = "rag"
    elif any(word in message for word in web_keywords):
        route = "web_search"
    else:
        route = "general"

    return {"route": route}


def general_chat_node(state: AssistantState):
    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are SmartAI Assistant. Explain clearly and simply. "
                    "Respond in the language used by the user when possible."
                ),
            },
            {"role": "user", "content": state["user_message"]},
        ],
        temperature=0.7,
        max_completion_tokens=1024,
    )

    return {"response": result.choices[0].message.content or ""}


def web_search_node(state: AssistantState):
    answer = web_search_response(state["user_message"])
    return {"response": answer}


def pdf_rag_node(state: AssistantState):
    chunks = state.get("pdf_chunks", [])
    embeddings = state.get("pdf_embeddings")

    if not chunks or embeddings is None:
        return {
            "response": (
                "Please upload a PDF first, then ask a question about it."
            )
        }

    query_embedding = create_embeddings([state["user_message"]])

    relevant_chunks = find_relevant_chunks(
        state["user_message"],
        chunks,
        embeddings,
        query_embedding,
        top_k=4,
    )

    if isinstance(relevant_chunks, str):
        context = relevant_chunks
    else:
        context = "\n\n".join(str(chunk) for chunk in relevant_chunks)

    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using the supplied PDF context. "
                    "If the answer is not present, say so clearly.\n\n"
                    f"PDF context:\n{context}"
                ),
            },
            {"role": "user", "content": state["user_message"]},
        ],
        temperature=0.3,
        max_completion_tokens=1024,
    )

    return {"response": result.choices[0].message.content or ""}


def choose_agent(state: AssistantState):
    return state["route"]


builder = StateGraph(AssistantState)

builder.add_node("router", router_node)
builder.add_node("general", general_chat_node)
builder.add_node("web_search", web_search_node)
builder.add_node("rag", pdf_rag_node)

builder.add_edge(START, "router")

builder.add_conditional_edges(
    "router",
    choose_agent,
    {
        "general": "general",
        "web_search": "web_search",
        "rag": "rag",
    },
)

builder.add_edge("general", END)
builder.add_edge("web_search", END)
builder.add_edge("rag", END)

assistant_graph = builder.compile()


if __name__ == "__main__":
    test_questions = [
        "What is machine learning?",
        "What are today's cricket news?",
        "According to my uploaded PDF, explain deep learning.",
    ]

    for question in test_questions:
        result = assistant_graph.invoke({
            "user_message": question,
            "route": "",
            "response": "",
            "pdf_chunks": [],
            "pdf_embeddings": None,
        })

        print("=" * 60)
        print("Question:", question)
        print("Selected route:", result["route"])
        print("Response:", result["response"][:500])
