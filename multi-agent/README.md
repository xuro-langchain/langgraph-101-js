# Multi Agents

In this module, we'll utilize LangGraph to create complex agents using a supervisor and subagent architecture. We'll utilize LangGraph prebuilts to see how we can speed up our development, and learn how to manually implement tools and nodes to create common architectures.

At the end, we'll have created a customer assistant that serves a digital music store!

Each module below can be run with the command npx tsx multi-agent/<module-name>.ts

## Module 1: Music ReAct Agent

This module is the ```multi-agent/music-subagent.ts``` file. It covers how to manually create a ReAct agent, including setting up tools, prompts and nodes. Our Music ReAct agent will handle all customer queries related to songs. The architecture diagram looks like
[Architecture](../images/music_subagent.png)

## Module 2: Invoice ReAct Agent

This module is the ```multi-agent/invoice-subagent.ts``` file. In this module, we'll create a second ReAct agent for handling invoice queries. We'll use LangGraph prebuilts to see how we can simplify our implementation. The architecture diagram looks like
[Architecture](../images/invoice_subagent.png)

## Module 3: Supervisor Agent

This module is the ```multi-agent/assistant-agent.ts``` file. In this module, we'll add a supervisor to route customer requests to the music or invoice agents, dependin on if the requests are song-related or invoice-related. This implements the common supervisor agent architecture, and will bring together earlier components to create a more complement customer service agent. The architecture diagram looks like [Architecture](../images/supervisor.png)

## Module 4: Supervisor with Human Oversight

This module is the ```multi-agent/hil-assistant.ts``` file. In this module, we'll add Human In the Loop to make our assistant more capable. We'll give ourselves the ability to accept user input, and verify customer information. The architecture diagram looks like [Architecture](../images/human_input.png)

## Module 5: Supervisor with Memory

This module is the ```multi-agent/memory-assistant.ts``` file. In this module, we'll add memory to personalize our agent to users over time. This will round out our music store assistant to give it realistic capabilities for customer service. The architecture diagram looks like [Architecture](../images/memory.png)