# RAG Agents

In this module, we'll cover the basics of LangGraph, and create a full-fledged RAG Agent.
We'll start with a simple RAG Graph, and add to it gradually to make it more complex.

## Module 1: Simple RAG

This module is the ```rag-agents/simple-rag.ts``` file. It covers the basics of LangGraph, including State, Edges, and Nodes. The diagram of the agent looks like 
[Architecture](../images/simple_rag.png)

## Module 2: Corrective RAG

This module is the ```rag-agents/corrective-rag.ts``` file. It covers more complicated graph
structures, and introduces the concepts of reflection into our RAG agent to improve its output
quality. The diagram of the agent looks like
[Architecture](../images/check_hallucinations.png)

## Module 3: Memory & HITL RAG

This module is the ```rag-agents/memory-hil-rag.ts``` file. It adds in memory and Human-in-the-Loop capabilities. This allows our agent to remember past messages and conversations, as well as defer to a human for oversight. The diagram of the agent looks like [Architecture](../images/hil_v2.png)