# LangGraph 101

Welcome to LangGraph 101! 

## Introduction
In this session, you will learn about the fundamentals of LangGraph through one of our notebooks. This is a condensed version of LangChain Academy, and is intended to be run in a session with a LangChain engineer.

## Context

At LangChain, we aim to make it easy to build LLM applications. One type of LLM application you can build is an agent. There’s a lot of excitement around building agents because they can automate a wide range of tasks that were previously impossible. 

In practice though, it is incredibly difficult to build systems that reliably execute on these tasks. As we’ve worked with our users to put agents into production, we’ve learned that more control is often necessary. You might need an agent to always call a specific tool first or use different prompts based on its state.

To tackle this problem, we’ve built [LangGraph](https://langchain-ai.github.io/langgraph/) — a framework for building agent and multi-agent applications. Separate from the LangChain package, LangGraph’s core design philosophy is to help developers add better precision and control into agent workflows, suitable for the complexity of real-world systems.

## Pre-work

### Clone the LangGraph 101 repo
```
git clone https://github.com/langchain-ai/langgraph-101.git
```

Navigate to setup.md and follow instructions there! If you run into issues with setting up the python environment or acquiring the necessary API keys due to any restrictions (ex. corporate policy), contact your LangChain representative and we'll find a work-around!

### Azure OpenAI Instructions

If you are using AzureOpenAI instead of OpenAI, there are a few things you need to do.

1. Set necessary environment variables in a .env file. Specifically, make sure you set
    - AZURE_OPENAI_API_KEY=
    - AZURE_OPENAI_ENDPOINT=
    - AZURE_OPENAI_API_VERSION=

2. Navigate to `models.py`, and uncomment the code for 
    - `AZURE_OPENAI_EMBEDDING_MODEL`= ...
    - `AZURE_OPENAI_GPT_4O`= ...

3. Navigate to `utils.py` and use AzureOpenAIEmbeddings instead of OpenAIEmbeddings

4. In the notebooks, use AzureOpenAI (code already provided in cells) where applicable, instead of OpenAI (default)