import { Client, Dataset } from "langsmith";
import { evaluate } from "langsmith/evaluation";
import { graph as multiAgent } from '../multi-agent/memory-assistant';
import { Command } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';

/* READ
Evaluating an agent's trajectory involves evaluating all the steps an agent took. 
The evaluator here is some function over the steps taken. 
Examples of evaluators include an exact match for each tool name in the sequence
or the number of "incorrect" steps taken.

- Input: User input to the overall agent 
- Output: A list of steps taken.
*/

const client = new Client();

// Define our Dataset
// --------------------------------------------------------------------------------
// We want the input to be the initial question to our assistant,
// and the output to be the trajectory of steps taken.
// --------------------------------------------------------------------------------
const examples = [
    {
        "question": "My customer ID is 1. What's my most recent purchase? and What albums does the catalog have by U2?",
        "trajectory": ["verify_info", "load_memory", "supervisor", "create_memory"],
    },
    {
        "question": "What songs do you have by U2?",
        "trajectory": ["verify_info", "human_input", "human_input", "verify_info", "human_input"],
    },
    {
        "question": "My name is Aaron Mitchell. My number associated with my account is +1 (204) 452-6452. I am trying to find the invoice number for my most recent song purchase. Could you help me with it?",
        "trajectory": ["verify_info", "load_memory", "supervisor", "create_memory"],
    },
    {
        "question": "Who recorded Wish You Were Here again? What other albums by them do you have?",
        "trajectory": ["verify_info", "human_input", "human_input", "verify_info", "human_input"],
    },
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
            inputs: examples.map(ex => ({ question: ex.question })),
            outputs: examples.map(ex => ({ trajectory: ex.trajectory })),
            datasetId: dataset.id
        });
        
        console.log(`Created dataset: ${name} with ${examples.length} examples`);
    } else {
        console.log(`Dataset already exists: ${name}`);
    }
}

// Create a Run Function 
// --------------------------------------------------------------------------------
// Run graph and track the trajectory it takes along with the final response.
// Note that here we must continue past the interrupt() 
// by supplying a Command({resume: ""}) to the graph.
// --------------------------------------------------------------------------------
async function runGraph(inputs: { question: string }): Promise<{ trajectory: string[] }> {
    const trajectory: string[] = [];
    const threadId = crypto.randomUUID();
    const config = { configurable: { thread_id: threadId, user_id: "10" } };

    // Run until interrupt 
    const stream1 = await multiAgent.stream(
        { messages: [new HumanMessage(inputs.question)] },
        config
    );
    
    for await (const chunk of stream1) {
        const nodeName = Object.keys(chunk)[0];
        if (nodeName && nodeName !== '__end__') {
            let pushNode = nodeName;
            if (nodeName === "__interrupt__") {
                pushNode = "human_input";
            }
            trajectory.push(pushNode);
        }
    }

    // Resume from interrupt
    const stream2 = await multiAgent.stream(
        new Command({ resume: "Not sayin anything." }),
        config
    );
    
    for await (const chunk of stream2) {
        const nodeName = Object.keys(chunk)[0];
        if (nodeName && nodeName !== '__end__') {
            let pushNode = nodeName;
            if (nodeName === "__interrupt__") {
                pushNode = "human_input";
            }
            trajectory.push(pushNode);
        }
    }
    
    return { trajectory };
}

// Create Evaluator Functions
// --------------------------------------------------------------------------------
// Evaluators that measure trajectory correctness.
// --------------------------------------------------------------------------------

async function evaluateExactMatch(
    run: any,
    example: any
): Promise<{ key: string; score: number }> {
    // Evaluate whether the trajectory exactly matches the expected output
    const isExactMatch = JSON.stringify(run.outputs.trajectory) === JSON.stringify(example.outputs.trajectory);
    
    return {
        key: "exact_match",
        score: isExactMatch ? 1.0 : 0.0
    };
}

async function evaluateExtraSteps(
    run: any,
    example: any
): Promise<{ key: string; score: number }> {
    // Evaluate the number of unmatched steps in the agent's output
    let i = 0;
    let j = 0;
    let unmatchedSteps = 0;

    while (i < example.outputs.trajectory.length && j < run.outputs.trajectory.length) {
        if (example.outputs.trajectory[i] === run.outputs.trajectory[j]) {
            i += 1; // Match found, move to the next step in reference trajectory
        } else {
            unmatchedSteps += 1; // Step is not part of the reference trajectory
        }
        j += 1; // Always move to the next step in outputs trajectory
    }

    // Count remaining unmatched steps in outputs beyond the comparison loop
    unmatchedSteps += run.outputs.trajectory.length - j;

    return {
        key: "unmatched_steps",
        score: unmatchedSteps,
    };
}

// Example usage
async function main() {
    // Create dataset first
    const datasetName = "LangGraph 101 JS: Trajectory Eval";
    await createDatasetIfNotExists(datasetName);
    
    // Run the evaluation using the LangSmith API
    await evaluate(runGraph, {
        data: datasetName,
        evaluators: [
            evaluateExtraSteps,
            evaluateExactMatch,
        ],
        experimentPrefix: "agent-o3mini-trajectory",
        maxConcurrency: 4,
    });
    
    console.log("Experiment Running! Check LangSmith for results.");
}

// Uncomment to run with the command npx tsx evals/trajectory.ts
main();