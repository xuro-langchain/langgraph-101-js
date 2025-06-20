import { graph as invoiceAgent } from './invoice-subagent';
import { graph as musicAgent } from './music-subagent';
import { llm } from './utils';
import { StateGraph, Annotation } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { createSupervisor } from "@langchain/langgraph-supervisor";

/* READ
After our 2 previous modules, we have two sub-agents that have different capabilities. 
How do we make sure customer tasks are appropriately routed between them? 

This is where the supervisor oversees the workflow,
 invoking appropriate subagents for relevant inquiries. 


A multi-agent architecture offers several key benefits:
- Specialization & Modularity – Each sub-agent is optimized for a specific task, 
  improving system accuracy 
- Flexibility – Agents can be quickly added, removed, or modified without affecting 
  the entire system

We will show how we can utilize the pre-built supervisor to quickly 
create the multi-agent architecture. 
*/

const supervisorPrompt = `You are an expert customer support assistant for a digital music store. 
You are dedicated to providing exceptional service and ensuring customer queries are answered thoroughly. 
You have a team of subagents that you can use to help answer queries from customers. 
Your primary role is to serve as a supervisor/planner for this multi-agent team that helps answer queries from customers. 

Your team is composed of two subagents that you can use to help answer the customer's request:
1. music_catalog_information_subagent: this subagent has access to user's saved music preferences. It can also retrieve information about the digital music store's music 
catalog (albums, tracks, songs, etc.) from the database. 
3. invoice_information_subagent: this subagent is able to retrieve information about a customer's past purchases or invoices 
from the database. 

Based on the existing steps that have been taken in the messages, your role is to generate the next subagent that needs to be called. 
This could be one step in an inquiry that needs multiple sub-agent calls.`;

// Define the state using Annotation
const SupervisorState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (currentState, updateValue) => [...(currentState || []), ...updateValue],
    default: () => [],
  }),
  customer_id: Annotation<string | null>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => null,
  }),
  loaded_memory: Annotation<string | null>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => null,
  }),
  remaining_steps: Annotation<any>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => null,
  }),
});

// --------------------------------------------------------------------------------
// New: Create supervisor workflow using prebuilt --------------------------------- 
// --------------------------------------------------------------------------------
const supervisorPrebuiltWorkflow = createSupervisor({
  agents: [invoiceAgent, musicAgent],
  outputMode: "last_message", // alternative is full_history
  llm: llm,
  prompt: supervisorPrompt,
  stateSchema: SupervisorState,
});

export const graph = supervisorPrebuiltWorkflow.compile();

// Example usage
async function main() {
  const question = "I need help with my recent invoice and also want to find some rock music";
  const stream = await graph.stream({ messages: [new HumanMessage(question)] });
  for await (const chunk of stream) {
    const nodeName = Object.keys(chunk)[0];
    const nodeData = chunk[nodeName as keyof typeof chunk];
    
    if (nodeData && typeof nodeData === 'object' && 'messages' in nodeData && Array.isArray(nodeData.messages)) {
      const lastMessage = nodeData.messages[nodeData.messages.length - 1];
      console.log(`\n--- ${nodeName} ---`);
      console.log(`Content: ${lastMessage.content}`);
      
      if ('tool_calls' in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length > 0) {
        console.log(`Tool calls: ${lastMessage.tool_calls.map((tc: any) => tc.name).join(', ')}`);
      }
    }
  }
}

// Uncomment to run
// main(); 