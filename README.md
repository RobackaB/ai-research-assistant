# AI Research Assistant (LangGraph)

## Overview

This project is a conversational AI research assistant built using LangGraph and LLMs.
It processes user queries, gathers relevant web information, and generates structured responses with sources.

## Features

* Two-step LLM workflow:

  * **Researcher node** → collects and summarizes web data
  * **Writer node** → generates final response
* Web search integration using Tavily API
* Source extraction and display
* Conversation history support for follow-up questions
* Clean CLI interface

## Tech Stack

* Python
* LangGraph
* LangChain
* OpenAI API
* Tavily API

## How It Works

1. User enters a topic or question
2. The system searches the web for relevant information
3. Research notes are generated
4. A final response is created and displayed
5. Sources are included at the end of the output

## Installation

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Run

```bash
python main.py
```

## Example Use

```text
You: Artificial Intelligence
```

The assistant will generate a structured response with sources.

## Notes

* This is a simplified AI workflow project for demonstration purposes
* Focus is on architecture (LLM + tools + state management)

## Author

Róbert Bajnok
