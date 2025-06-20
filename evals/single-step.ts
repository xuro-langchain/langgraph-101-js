import { Client, Dataset } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { HumanMessage } from '@langchain/core/messages';
import { supervisorPrebuiltWorkflow } from '../multi-agent/assistant-agent';

/* READ
Single Step Evaluations
Agents generally perform multiple actions. 
While it is useful to evaluate them end-to-end, 
it can also be useful to evaluate these individual actions, 
similar to the concept of unit testing in software development. 

This generally involves evaluating a single step of the agent 
- the LLM call where it decides what to do.

- Input: Input to a single step 
- Output: Output of that step, which is usually the LLM response
*/

const client = new Client();

// Define our Dataset
// --------------------------------------------------------------------------------
// We want the input to be the messages sent to the supervisor,
// and the output to be the route decision made by the supervisor.
// --------------------------------------------------------------------------------
const examples = [
    {
        "messages": "My customer ID is 1. What's my most recent purchase? and What albums does the catalog have by U2?", 
        "route": 'transfer_to_invoice_information_subagent'
    },
    {
        "messages": "What songs do you have by U2?", 
        "route": 'transfer_to_music_catalog_subagent'
    },
    {
        "messages": "My name is Aaron Mitchell. My number associated with my account is +1 (204) 452-6452. I am trying to find the invoice number for my most recent song purchase. Could you help me with it?", 
        "route": 'transfer_to_invoice_information_subagent'
    },
    {
        "messages": "Who recorded Wish You Were Here again? What other albums by them do you have?", 
        "route": 'transfer_to_music_catalog_subagent'
    }
];

// Create dataset if it doesn't exist
async function createDatasetIfNotExists(name: string) {
    // Check if dataset exists
    let existingDataset: Dataset | null = null;
    for await (const dataset of client.listDatasets({ datasetName: name })) {
        if (dataset.name === name) { existingDataset = dataset; break; }
    }
    
    if (!existingDataset) {
        // Create the dataset
        const dataset = await client.createDataset(name);
        
        // Create examples
        await client.createExamples({
            inputs: examples.map(ex => ({ messages: ex.messages })),
            outputs: examples.map(ex => ({ route: ex.route })),
            datasetId: dataset.id
        });
        
        console.log(`Created dataset: ${name} with ${examples.length} examples`);
    } else {
        console.log(`Dataset already exists: ${name}`);
    }
}

// Create a Run Function 
// --------------------------------------------------------------------------------
// We only need to evaluate the supervisor routing step, 
// so let's add a breakpoint right after the supervisor step.
// --------------------------------------------------------------------------------


async function runSupervisorRouting(inputs: { messages: string }): Promise<{ route: string }> {
    const threadId = crypto.randomUUID();
    const configuration = { 
        configurable: { 
            thread_id: threadId, 
            user_id: "10",
        },
    };
    
    const supervisorGraph = supervisorPrebuiltWorkflow.compile({
        interruptBefore: ["music_catalog_subagent", "invoice_information_subagent"]
    });
    // Use the actual multi-agent graph to get supervisor routing
    const result = await supervisorGraph.invoke(
        { messages: [new HumanMessage(inputs.messages)] },
        configuration
    );
    
    // Extract the route from the supervisor's decision
    // The supervisor node should be the one that decides routing
    const lastMessage = result.messages[result.messages.length - 1];
    const route = lastMessage.name || "error";    
    return { route };
}

// Create an Evaluator Function
// --------------------------------------------------------------------------------
// An evaluator that checks if the agent chose the correct route.
// --------------------------------------------------------------------------------
async function correct(
    run: any,
    example: any
): Promise<{ key: string; score: number }> {
    // Check if the agent chose the correct route
    const isCorrect = run.outputs.route === example.outputs.route;
    
    return {
        key: "route_correctness",
        score: isCorrect ? 1.0 : 0.0
    };
}

// Example usage
async function main() {
    // Create dataset first
    const datasetName = "LangGraph 101 JS: Single-Step";
    await createDatasetIfNotExists(datasetName);
    
    // Run the evaluation using the LangSmith API
    await evaluate(runSupervisorRouting, {
        data: datasetName,
        evaluators: [
            correct,
            // can add multiple evaluators here
        ],
        experimentPrefix: "agent-o3mini-singlestep",
        maxConcurrency: 5,
    });
    
    console.log("Experiment Running! Check LangSmith for results.");
}

// Uncomment to run with the command npx ts-node evals/single-step.ts
main();