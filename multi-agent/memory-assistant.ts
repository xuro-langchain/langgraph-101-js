import { llm } from './utils';
import { StateGraph, Annotation, Command } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { InMemoryStore } from "@langchain/langgraph";

import { z } from 'zod';
import { MemorySaver } from '@langchain/langgraph';

// Import functions from hil-assistant.ts to reuse in graph
import { 
  verifyInfo, 
  humanInput, 
  shouldInterrupt, 
  supervisorPrebuilt
} from './hil-assistant';

import { SupervisorState as AssistantState } from './assistant-agent';

/* READ
Now that we have created an agent workflow that includes verification and execution, 
let's take it a step further. 

Long term memory lets you store and recall information between conversations. 
We'll instantiate a long term memory store in this step. 
*/
const store = new InMemoryStore(); // or a database-backed store

// ------------------------------------------------------------------------------------------------------------
// New: Long Term Memory Nodes -------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
/*
In this step, we will add 2 nodes: 
- **load_memory** node that loads from the long term memory store
- **create_memory** node that saves any music interests that the customer has shared about themselves 
*/
// --- User Profile Type ---
export interface UserProfile {
  customer_id: string;
  music_preferences: string[];
}

// --- Helper to format user memory ---
function formatUserMemory(userProfile: UserProfile | undefined): string {
  if (!userProfile) return '';
  if (userProfile.music_preferences && userProfile.music_preferences.length > 0) {
    return `Music Preferences: ${userProfile.music_preferences.join(', ')}`;
  }
  return '';
}

// --- Node: Load memory ---
async function loadMemory(state: GraphState): Promise<Partial<GraphState>> {
  const userId = state.customer_id;
  if (!userId) return { loaded_memory: '' };
  
  const namespace = [userId, "memory_profile"];
  const existingMemory = await store.get(namespace, "user_memory");
  
  let userProfile: UserProfile | undefined;
  if (existingMemory && existingMemory.value) {
    userProfile = existingMemory.value as UserProfile;
  }
  
  const formatted = formatUserMemory(userProfile);
  return { loaded_memory: formatted };
}

// --- Zod schema for structured output ---
const UserProfileSchema = z.object({
  customer_id: z.string(),
  music_preferences: z.array(z.string())
});

// --- Prompt for memory creation/update ---
const createMemoryPrompt = `
You are an expert analyst that is observing a conversation that has taken place between a customer and a customer support assistant. The customer support assistant works for a digital music store, and has utilized a multi-agent team to answer the customer's request. 
You are tasked with analyzing the conversation that has taken place between the customer and the customer support assistant, and updating the memory profile associated with the customer. The memory profile may be empty. If it's empty, you should create a new memory profile for the customer.

You specifically care about saving any music interest the customer has shared about themselves, particularly their music preferences to their memory profile.

To help you with this task, I have attached the conversation that has taken place between the customer and the customer support assistant below, as well as the existing memory profile associated with the customer that you should either update or create. 

The customer's memory profile should have the following fields:
- customer_id: the customer ID of the customer
- music_preferences: the music preferences of the customer

These are the fields you should keep track of and update in the memory profile. If there has been no new information shared by the customer, you should not update the memory profile. It is completely okay if you do not have new information to update the memory profile with. In that case, just leave the values as they are.

*IMPORTANT INFORMATION BELOW*

The conversation between the customer and the customer support assistant that you should analyze is as follows:
{conversation}

The existing memory profile associated with the customer that you should either update or create based on the conversation is as follows:
{memory_profile}

Ensure your response is an object that has the following fields:
- customer_id: the customer ID of the customer
- music_preferences: the music preferences of the customer

For each key in the object, if there is no new information, do not update the value, just keep the value that is already there. If there is new information, update the value. 

Take a deep breath and think carefully before responding.
`;

// --- Node: Create or update memory ---
async function createOrUpdateMemory(state: GraphState): Promise<Partial<GraphState>> {
  const userId = state.customer_id;
  if (!userId) return {};
  
  const namespace = [userId, "memory_profile"];
  const existingMemory = await store.get(namespace, "user_memory");
  
  let existingProfile: UserProfile | undefined;
  if (existingMemory && existingMemory.value) {
    existingProfile = existingMemory.value as UserProfile;
  }
  
  const formattedMemory = formatUserMemory(existingProfile);

  // Prepare the conversation string (you may want to format this better)
  const conversation = (state.messages || [])
    .map(m => `${m.constructor.name}: ${m.content}`)
    .join('\n');

  const systemPrompt = createMemoryPrompt
    .replace('{conversation}', conversation)
    .replace('{memory_profile}', formattedMemory);

  const structuredLLM = llm.withStructuredOutput(UserProfileSchema);
  const updatedProfile = await structuredLLM.invoke([
    new SystemMessage(systemPrompt)
  ]);

  // Store the updated profile
  await store.put(namespace, "user_memory", updatedProfile);

  return { loaded_memory: formatUserMemory(updatedProfile) };
}

// --- State definition (using the same state as hil-assistant) ---
const State = AssistantState;
type GraphState = typeof State.State;

// --- Build the multi-agent with long-term memory ---
const multiAgentMemory = new StateGraph(State);

// Add nodes
multiAgentMemory
.addNode("verify_info", verifyInfo)
.addNode("human_input", humanInput)
.addNode("load_memory", loadMemory)
.addNode("supervisor", supervisorPrebuilt)
.addNode("create_memory", createOrUpdateMemory)
.addEdge("__start__", "verify_info")
.addConditionalEdges(
  "verify_info",
  shouldInterrupt,
  {
    "continue": "load_memory",
    "interrupt": "human_input",
  }
)
.addEdge("human_input", "verify_info")
.addEdge("load_memory", "supervisor")
.addEdge("supervisor", "create_memory")
.addEdge("create_memory", "__end__");

// Add memory support for checkpointer
const memory = new MemorySaver();
export const graph = multiAgentMemory.compile({ 
  name: "multi_agent_memory", 
  checkpointer: memory,
});

// --- Example usage ---
async function main() {
  const question = "My customer ID is 1. How much was my most recent purchase? What albums do you have by the Rolling Stones?";
  
  // Create configuration with required thread_id for MemorySaver
  const thread_id = crypto.randomUUID();
  const config = { configurable: { thread_id } };
  
  const stream = await graph.stream({ messages: [new HumanMessage(question)] }, config);
  
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

  // Check the memory store after processing, and see that our preferences are saved
  console.log('\n--- Memory Store Check ---');
  const userId = "1"; // Since we used customer ID 1 in the question
  const namespace = [userId, "memory_profile"];
  const memory = await store.get(namespace, "user_memory");
  if (memory && memory.value) {
    console.log(`Customer ${userId}:`, memory.value);
  } else {
    console.log(`No memory found for customer ${userId}`);
  }
}

// Uncomment to run, use command: npx tsx multi-agent/memory-assistant.ts
// main();