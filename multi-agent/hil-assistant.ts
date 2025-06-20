import { supervisorPrebuiltWorkflow, GraphState, SupervisorState } from './assistant-agent';
import { llm, getEngineForChinookDb, queryChinookDb } from './utils';
import { StateGraph, Annotation, Command } from '@langchain/langgraph';
import { interrupt } from "@langchain/langgraph";
import { MemorySaver } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { createSupervisor } from "@langchain/langgraph-supervisor";
import { z } from 'zod';

/* READ
After our first 3 modules, we have seen how we can build a multi-agent system from scratch. 
We currently invoke our multi-agent system with a customer ID as the customer identifier, 
but realistically, we may not always have access to the customer identity. 
To solve this, we want to **first verify the customer information** before 
executing their inquiry with our supervisor agent. 

In this step, we will be showing a simple implementation of such a node, 
using **human-in-the-loop** to prompt the customer to provide their account information. 
*/

// Initialize database connection
let db: any;

async function initializeDb() {
  if (!db) {
    db = await getEngineForChinookDb();
  }
  return db;
}


// ------------------------------------------------------------------------------------------------------------
// New: Human In the Loop Nodes -------------------------------------------------------------
// ------------------------------------------------------------------------------------------------------------
/*
In this step, we will write two nodes: 
- **verify_info** node that verifies account information 
- **human_input** node that prompts user to provide additional information 

ChatModels support attaching a structured data schema to adhere response to. 
This is useful in scenarios like extracting information or categorizing. 
*/

// Schema for parsing user-provided account information
const UserInputSchema = z.object({
  identifier: z.string().describe("Identifier, which can be a customer ID, email, or phone number.")
});

// Helper function to get customer ID from identifier
export async function getCustomerIdFromIdentifier(identifier: string): Promise<string | null> {
  const dbInstance = await initializeDb();
  
  if (/^\d+$/.test(identifier)) {
    return identifier;
  } else if (identifier.startsWith('+')) {
    const query = `SELECT CustomerId FROM Customer WHERE Phone = '${identifier}';`;
    const result = await queryChinookDb(dbInstance, query);
    if (result && result.length > 0) {
      return result[0].CustomerId.toString();
    }
  } else if (identifier.includes('@')) {
    const query = `SELECT CustomerId FROM Customer WHERE Email = '${identifier}';`;
    const result = await queryChinookDb(dbInstance, query);
    if (result && result.length > 0) {
      return result[0].CustomerId.toString();
    }
  }
  return null;
}

// Node: Verify customer information
export async function verifyInfo(state: GraphState): Promise<Partial<GraphState>> {
  if (state.customer_id === null) {
    const systemInstructions = `You are a music store agent, where you are trying to verify the customer identity 
    as the first step of the customer support process. 
    Only after their account is verified, you would be able to support them on resolving the issue. 
    In order to verify their identity, one of their customer ID, email, or phone number needs to be provided.
    If the customer has not provided their identifier, please ask them for it.
    If they have provided the identifier but cannot be found, please ask them to revise it.`;

    const userInput = state.messages[state.messages.length - 1];
    
    // Parse for customer ID using structured output
    const structuredLLM = llm.withStructuredOutput(UserInputSchema);
    
    try {
      const parsedInfo = await structuredLLM.invoke([
        new SystemMessage("You are a customer service representative responsible for extracting customer identifier. Only extract the customer's account information from the message history. If they haven't provided the information yet, return an empty string for the identifier"),
        userInput
      ]);
      
      let customerId: string | null = null;
      if (parsedInfo.identifier) {
        customerId = await getCustomerIdFromIdentifier(parsedInfo.identifier);
      }
      
      if (customerId) {
        const intentMessage = new SystemMessage(
          `Thank you for providing your information! I was able to verify your account with customer id ${customerId}.`
        );
        return {
          customer_id: customerId as string,
          messages: [intentMessage]
        };
      } else {
        const response = await llm.invoke([
          new SystemMessage(systemInstructions),
          ...state.messages
        ]);
        return { messages: [response] };
      }
    } catch (error) {
      // Fallback to regular LLM if structured output fails
      const response = await llm.invoke([
        new SystemMessage(systemInstructions),
        ...state.messages
      ]);
      return { messages: [response] };
    }
  }
  return {};
}

// Node: Human input (interrupt point)
export async function humanInput(state: GraphState): Promise<Partial<GraphState>> {
  // Use the interrupt mechanism to ask for customer information
  const userInput = await interrupt("Please provide your customer information (ID, email, or phone number).");
  return { messages: [userInput] };
}

// Conditional edge: should interrupt
export function shouldInterrupt(state: GraphState): string {
  if (state.customer_id !== null) {
    return "continue";
  } else {
    return "interrupt";
  }
}

// Compile supervisor
export const supervisorPrebuilt = supervisorPrebuiltWorkflow.compile({ name: "multi-agent-entry" });

// Build the multi-agent verification workflow
const multiAgentVerify = new StateGraph(SupervisorState);

// Add nodes
multiAgentVerify
.addNode("verify_info", verifyInfo)
.addNode("human_input", humanInput)
.addNode("supervisor", supervisorPrebuilt)
.addEdge("__start__", "verify_info")
.addConditionalEdges(
  "verify_info",
  shouldInterrupt,
  {
    "continue": "supervisor",
    "interrupt": "human_input",
  }
)
.addEdge("human_input", "verify_info")
.addEdge("supervisor", "__end__");

// Add memory support for checkpointer
const memory = new MemorySaver();
export const graph = multiAgentVerify.compile({ 
  name: "multi_agent_verify",
  checkpointer: memory 
});

async function printStream(stream: AsyncGenerator<any, any, any>) {
  for await (const chunk of stream) {
    const nodeName = Object.keys(chunk)[0];
    const nodeData = chunk[nodeName as keyof typeof chunk];
    
    if (nodeData && typeof nodeData === 'object' && 'messages' in nodeData && Array.isArray(nodeData.messages)) {
      const lastMessage = nodeData.messages[nodeData.messages.length - 1];
      console.log(`\n--- ${nodeName} ---`);
      // Handle string or object message
      if (typeof lastMessage === 'string') {
        console.log(`Content: ${lastMessage}`);
      } else {
        console.log(`Content: ${lastMessage.content}`);
        if ('tool_calls' in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length > 0) {
          console.log(`Tool calls: ${lastMessage.tool_calls.map((tc: any) => tc.name).join(', ')}`);
        }
      }
    }
  }
}

// Example usage
async function main() {
  const thread_id = crypto.randomUUID();
  const config = { configurable: { thread_id } };
  
  let question = "How much was my most recent purchase?";
  let first_stream = await graph.stream({ messages: [new HumanMessage(question)] }, config);
  await printStream(first_stream);

  let feedback = "My phone number is +55 (12) 3923-5555.";
  let second_stream = await graph.stream(new Command({ resume: feedback }), config);
  await printStream(second_stream);

}

// Uncomment to run, use command: npx tsx multi-agent/hil-assistant.ts
// main(); 