from __future__ import annotations

import json
import os
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from openai import BadRequestError


MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-5-nano")


class ResearchState(TypedDict):
    user_request: str
    research_notes: str
    final_response: str
    conversation_history: list[str]
    sources: list[str]


def _to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts).strip()
    return str(content).strip()


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=MODEL_NAME,
        use_responses_api=True,
    )


def _format_history(history: list[str]) -> str:
    if not history:
        return "No previous conversation."
    return "\n".join(history)


def _truncate(text: str, limit: int = 1000) -> str:
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _extract_sources(search_results: Any) -> list[str]:
    sources: list[str] = []
    if isinstance(search_results, list):
        for item in search_results:
            if isinstance(item, dict):
                url = item.get("url")
                if isinstance(url, str) and url not in sources:
                    sources.append(url)
    elif isinstance(search_results, dict):
        results = search_results.get("results")
        if isinstance(results, list):
            for item in results:
                if isinstance(item, dict):
                    url = item.get("url")
                    if isinstance(url, str) and url not in sources:
                        sources.append(url)
    return sources


def researcher(state: ResearchState) -> ResearchState:
    user_request = state["user_request"]
    history_text = _format_history(state["conversation_history"])

    search_tool = TavilySearch(
        max_results=5,
        topic="general",
    )

    search_results = search_tool.invoke(
        {"query": f"Latest information and facts about: {user_request}"}
    )
    sources = _extract_sources(search_results)

    llm = _build_llm()

    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are a Researcher. Analyze web results and produce concise, structured research notes. Focus on key facts, trends, and useful insights. Use conversation history if needed.",
                ),
                (
                    "human",
                    f"Conversation history:\n{history_text}\n\n"
                    f"User request:\n{user_request}\n\n"
                    f"Search results:\n{json.dumps(search_results, indent=2, default=str)}",
                ),
            ]
        )
    except BadRequestError as exc:
        raise RuntimeError(
            f"OpenAI request failed for model '{MODEL_NAME}'. Check API access."
        ) from exc

    return {
        "research_notes": _to_text(response.content),
        "sources": sources,
    }


def writer(state: ResearchState) -> ResearchState:
    llm = _build_llm()
    history_text = _format_history(state["conversation_history"])

    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "You are a Writer. Turn research notes into a clear and useful response. Adapt format to the request: short answer or structured markdown when appropriate.",
                ),
                (
                    "human",
                    f"Conversation history:\n{history_text}\n\n"
                    f"User request:\n{state['user_request']}\n\n"
                    f"Research notes:\n{state['research_notes']}\n\n"
                    f"Sources:\n{json.dumps(state['sources'], indent=2)}",
                ),
            ]
        )
    except BadRequestError as exc:
        raise RuntimeError(
            f"OpenAI request failed for model '{MODEL_NAME}'. Check API access."
        ) from exc

    final_text = _to_text(response.content)

    if state["sources"]:
        final_text += "\n\nSources:\n" + "\n".join(f"- {source}" for source in state["sources"])

    return {
        "final_response": final_text,
    }


def build_graph():
    graph_builder = StateGraph(ResearchState)

    graph_builder.add_node("researcher", researcher)
    graph_builder.add_node("writer", writer)

    graph_builder.add_edge(START, "researcher")
    graph_builder.add_edge("researcher", "writer")
    graph_builder.add_edge("writer", END)

    return graph_builder.compile()


def run_chat() -> None:
    graph = build_graph()
    conversation_history: list[str] = []

    print("AI Research Assistant is ready.")
    print("Enter a topic or question. Type 'exit', 'quit', or 'stop' to end.")

    while True:
        user_input = input("\nYou: ").strip()

        if not user_input:
            print("Please enter a topic or question.")
            continue

        if user_input.lower() in {"exit", "quit", "stop"}:
            print("Ending conversation.")
            break

        try:
            result = graph.invoke(
                {
                    "user_request": user_input,
                    "research_notes": "",
                    "final_response": "",
                    "conversation_history": conversation_history,
                    "sources": [],
                }
            )
        except Exception as exc:
            print(f"\nAssistant error: {exc}")
            continue

        final_response = result["final_response"]

        print("\nAssistant:\n")
        print(final_response)

        conversation_history.extend(
            [
                f"User: {user_input}",
                f"Assistant: {_truncate(final_response)}",
            ]
        )

        conversation_history = conversation_history[-6:]


if __name__ == "__main__":
    load_dotenv()
    load_dotenv("env.env")

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY.")

    if not os.getenv("TAVILY_API_KEY"):
        raise RuntimeError("Missing TAVILY_API_KEY.")

    run_chat()