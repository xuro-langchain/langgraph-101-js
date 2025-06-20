import { graph as invoiceAgent } from './invoice-subagent';
import { graph as musicAgent } from './music-subagent';
import { llm } from './utils';
import { v4 as uuidv4 } from 'uuid';
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

// Helper reducer to combine and deduplicate messages when using supervisor
function addMessages(left: BaseMessage[], right: BaseMessage[]): BaseMessage[] {
  let leftCopy = Array.isArray(left) ? left.slice() : [left];
  let rightCopy = Array.isArray(right) ? right.slice() : [right];

  // Assign missing ids
  leftCopy.forEach(m => {
      if (!m.id) {
          m.id = uuidv4();
      }
  });
  rightCopy.forEach(m => {
      if (!m.id) {
          m.id = uuidv4();
      }
  });

  let leftIdxById: Record<string, number> = {};
  leftCopy.forEach((m, i) => {
      leftIdxById[m.id!] = i;
  });

  let merged = leftCopy.slice();
  rightCopy.forEach(m => {
      let existingIdx = leftIdxById[m.id!];
      if (existingIdx !== undefined) {
          merged[existingIdx] = m;
      } else {
          merged.push(m);
      }
  });

  return merged;
}

// Define the state using Annotation
export const SupervisorState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: addMessages,
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
export const supervisorPrebuiltWorkflow = createSupervisor({
  agents: [invoiceAgent, musicAgent],
  outputMode: "last_message", // alternative is full_history
  llm: llm,
  prompt: supervisorPrompt,
  stateSchema: SupervisorState,
});

export const graph = supervisorPrebuiltWorkflow.compile();
export type GraphState = typeof SupervisorState.State;

// Example usage
async function main() {
  const question = "My customer ID is 1. How much was my most recent purchase? What albums do you have by U2";
  const stream = await graph.stream({ messages: [new HumanMessage(question)] });
  for await (const chunk of stream) {
    const nodeName = Object.keys(chunk)[0];
    const nodeData = chunk[nodeName as keyof typeof chunk];
    
    if (nodeData && typeof nodeData === 'object' && 'messages' in nodeData && Array.isArray(nodeData.messages)) {
      console.log(`\n--- ${nodeName} ---`);
      
      // Print all messages in this chunk
      for (const message of nodeData.messages) {
        if (message.getType() === 'tool') {
          console.log(`Tool Result: ${message.content}`);
        } else if (message.getType() === 'ai') {
          console.log(`AI Response: ${message.content}`);
          if ('tool_calls' in message && Array.isArray(message.tool_calls) && message.tool_calls.length > 0) {
            console.log(`Tool calls: ${message.tool_calls.map((tc: any) => tc.name).join(', ')}`);
          }
        } else if (message.getType() === 'human') {
          console.log(`Human Input: ${message.content}`);
        }
      }
    }
  }
}

// Uncomment to run, use command: npx tsx multi-agent/assistant-agent.ts
// main(); 