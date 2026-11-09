from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from core.groq_client import MODEL_NAME, client
from rag.embeddings import create_embeddings
from rag.retriever import find_relevant_chunks
from search.web_search import web_search_response


class AssistantState(TypedDict):
    user_message: str
    messages: list[dict[str, Any]]
    web_search_enabled: bool
    pdf_chunks: list[str]
    pdf_embeddings: Any
    route: str
    response: str


def router_node(state: AssistantState) -> dict[str, str]:
    """Choose the appropriate agent for the user's current question."""
    message = state["user_message"].lower()

    pdf_keywords = [
        "uploaded pdf",
        "my pdf",
        "this pdf",
        "the pdf",
        "in the pdf",
        "from the pdf",
        "according to the pdf",
        "uploaded document",
        "my document",
        "this document",
        "the document",
        "in the document",
        "from the document",
        "according to the document",
        "uploaded file",
        "this file",
        "in this file",
        "from this file",
        "summarize this document",
        "summarise this document",
        "summarize this pdf",
        "summarise this pdf",
    ]

    # An enabled Web Search toggle takes priority.
    if state.get("web_search_enabled", False):
        route = "web_search"
    # Use PDF RAG only when the question refers to a PDF/document.
    elif any(keyword in message for keyword in pdf_keywords):
        route = "rag"
    # Otherwise, answer as a normal general-chat question.
    else:
        route = "general"

    return {"route": route}


def general_chat_node(state: AssistantState) -> dict[str, str]:
    """Generate a normal conversational response with Groq."""
    messages = state.get("messages", [])

    if not messages:
        messages = [{"role": "user", "content": state["user_message"]}]

    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
        max_completion_tokens=1024,
    )

    return {"response": result.choices[0].message.content or ""}


def web_search_node(state: AssistantState) -> dict[str, str]:
    """Use the project's existing web-search workflow."""
    answer = web_search_response(state["user_message"])
    return {"response": answer or "I couldn't find a useful web-search result."}


def pdf_rag_node(state: AssistantState) -> dict[str, str]:
    """Retrieve relevant PDF chunks and answer from the retrieved context."""
    chunks = state.get("pdf_chunks") or []
    embeddings = state.get("pdf_embeddings")

    if not chunks or embeddings is None:
        return {
            "response": "Please upload a PDF first, then ask a question about its contents."
        }

    query_embedding = create_embeddings([state["user_message"]])[0]
    relevant_chunks = find_relevant_chunks(
        state["user_message"],
        chunks,
        embeddings,
        query_embedding,
        top_k=4,
    )

    if not relevant_chunks:
        return {
            "response": "I couldn't find relevant information in the PDF to answer that question."
        }

    context_parts = []
    for item in relevant_chunks:
        if isinstance(item, dict):
            context_parts.append(str(item.get("chunk", item)))
        else:
            context_parts.append(str(item))

    context = "\n\n".join(context_parts)

    # Keep conversation history, but use a PDF-specific system instruction.
    history = [
        message
        for message in state.get("messages", [])
        if message.get("role") != "system"
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are SmartAI Assistant. Answer the user's question using "
                "the PDF context below. If the answer is not supported by the "
                "context, say so clearly. Respond in the user's language "
                "(English, Tamil, or Thanglish) as appropriate.\n\n"
                f"PDF CONTEXT:\n{context}"
            ),
        },
        *history,
    ]

    result = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.3,
        max_completion_tokens=1024,
    )

    return {"response": result.choices[0].message.content or ""}


def choose_agent(state: AssistantState) -> str:
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
        ("What is machine learning?", False),
        ("What are today's cricket news?", True),
        ("According to my uploaded PDF, explain deep learning.", False),
    ]

    for question, web_enabled in test_questions:
        result = assistant_graph.invoke(
            {
                "user_message": question,
                "messages": [{"role": "user", "content": question}],
                "web_search_enabled": web_enabled,
                "pdf_chunks": [],
                "pdf_embeddings": None,
                "route": "",
                "response": "",
            }
        )
        print("=" * 60)
        print("Question:", question)
        print("Selected route:", result["route"])
        print("Response:", result["response"][:400])
