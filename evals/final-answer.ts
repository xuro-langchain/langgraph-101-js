import { Client, Dataset } from "langsmith";
import { graph as multiAgent } from '../multi-agent/memory-assistant';
import { evaluate } from "langsmith/evaluation";
import { llm } from '../multi-agent/utils';
import { Command } from '@langchain/langgraph';
import { HumanMessage } from '@langchain/core/messages';
import { z } from 'zod';

/* READ
Evaluations are made up of three components:

1. A **dataset test** with inputs and expected outputs.
2. An **application or target function** that defines what you are evaluating, taking in inputs and returning the application output
3. **Evaluators** that score your target function's outputs.

Final Response Evaluations

One way to evaluate an agent is to assess its overall performance on a task. 
This basically involves treating the agent as a black box and 
simply evaluating whether or not it gets the job done.

- Input: User input 
- Output: The agent's final response.
*/

const client = new Client();

// Define our Dataset
// --------------------------------------------------------------------------------
// We want the input to be the initial question to our assistant,
// and the output to be the final response from the assistant.
// --------------------------------------------------------------------------------
const examples = [
    {
        "question": "My name is Aaron Mitchell. My number associated with my account is +1 (204) 452-6452. I am trying to find the invoice number for my most recent song purchase. Could you help me with it?",
        "response": "The Invoice ID of your most recent purchase was 342.",
    },
    {
        "question": "I'd like a refund.",
        "response": "I need additional information to help you with the refund. Could you please provide your customer identifier so that we can fetch your purchase history?",
    },
    {
        "question": "Who recorded Wish You Were Here again?",
        "response": "Wish You Were Here is an album by Pink Floyd",
    },
    { 
        "question": "What albums do you have by Coldplay?",
        "response": "There are no Coldplay albums available in our catalog at the moment.",
    },
];

// Create dataset if it doesn't exist
async function createDatasetIfNotExists(name: string) {
    // Check if dataset exists
    let existingDataset : Dataset | null = null;
    for await (const dataset of client.listDatasets({ datasetName: name })) {
        if (dataset.name === name) { existingDataset = dataset; break; }
    }
    
    if (!existingDataset) {
      // Create the dataset
      const dataset = await client.createDataset(name);
      
      // Create examples - using the current working approach
      await client.createExamples({
        inputs: examples.map(ex => ({ question: ex.question })),
        outputs: examples.map(ex => ({ response: ex.response })),
        datasetId: dataset.id
      });
      
      console.log(`Created dataset: ${name} with ${examples.length} examples`);
    } else {
      console.log(`Dataset already exists: ${name}`);
    }
}

// Create a Run Function 
// --------------------------------------------------------------------------------
// Now, let's define how to run our graph.
// Note that here we must continue past the interrupt() 
// by supplying a Command({resume: ""}) to the graph. */
// --------------------------------------------------------------------------------
async function runGraph(inputs: { question: string }): Promise<{ response: string }> {
    // Run graph and track the final response
    // Creating configuration 
    const threadId = crypto.randomUUID();
    const configuration = { configurable: { thread_id: threadId } };

    // Invoke graph until interrupt 
    const result = await multiAgent.invoke({ 
        messages: [new HumanMessage(inputs.question)]
    }, configuration);
    
    // Proceed from human-in-the-loop 
    const finalResult = await multiAgent.invoke(
        new Command({ resume: "My customer ID is 10" }), 
        { configurable: { thread_id: threadId, user_id: "10" } }
    );
    
    const lastMessage = finalResult.messages[finalResult.messages.length - 1];
    const content = typeof lastMessage.content === 'string' 
        ? lastMessage.content 
        : JSON.stringify(lastMessage.content);
    
    return { response: content };
}

// Create an Evaluator Function
// --------------------------------------------------------------------------------
// An evaluator is a function that measures some kind of metric based on the
// dataset input, ground truth output, and your applications output. 

// During an evaluation, your application will run on the dataset inputs and generate outputs.
// The evaluator will compare your application's output 
// to the ground truth (reference) output in your dataset.
// Using an LLM to make the comparison between your agent's output and the reference outpus
// is a common appraoch for handling open ended metrics like correctness or relevancy.
// --------------------------------------------------------------------------------


// Custom definition of LLM-as-judge instructions
const graderInstructions = `You are a teacher grading a quiz.

You will be given a QUESTION, the GROUND TRUTH (correct) RESPONSE, and the STUDENT RESPONSE.

Here is the grade criteria to follow:
(1) Grade the student responses based ONLY on their factual accuracy relative to the ground truth answer.
(2) Ensure that the student response does not contain any conflicting statements.
(3) It is OK if the student response contains more information than the ground truth response, as long as it is factually accurate relative to the ground truth response.

Correctness:
True means that the student's response meets all of the criteria.
False means that the student's response does not meet all of the criteria.

Explain your reasoning in a step-by-step manner to ensure your reasoning and conclusion are correct.`;

// LLM-as-judge output schema
const GradeSchema = z.object({
    reasoning: z.string().describe("Explain your reasoning for whether the actual response is correct or not."),
    is_correct: z.boolean().describe("True if the student response is mostly or exactly correct, otherwise False.")
});


// Judge LLM
const graderLLM = llm.withStructuredOutput(GradeSchema);

// Evaluator function
async function finalAnswerCorrect(
  run: any,
  example: any
): Promise<{ key: string; score: number; comment: string }> {
  // Evaluate if the final response is equivalent to reference response
  const user = `QUESTION: ${example.inputs.question}
  GROUND TRUTH RESPONSE: ${example.outputs.response}
  STUDENT RESPONSE: ${run.outputs.response}`;

  const grade = await graderLLM.invoke([
    { role: "system", content: graderInstructions }, 
    { role: "user", content: user }
  ]);
  
  return { 
    key: "correctness",
    score: grade.is_correct ? 1.0 : 0.0,
    comment: grade.reasoning
  };
}

// Example usage
async function main() {
  // Create dataset first
  const datasetName = "LangGraph 101 JS: Final Response";
  await createDatasetIfNotExists(datasetName);
  
  // Create and run the experiment
  const experimentName = "multi-agent-final-response-eval";
  
  // Run the evaluation using the correct LangSmith API
  await evaluate(runGraph, {
    data: datasetName,
    evaluators: [
      finalAnswerCorrect,
      // can add multiple evaluators here
    ],
    experimentPrefix: "gpt-4o-mini",
    maxConcurrency: 2,
  });
  
  console.log("Experiment Running! Check LangSmith for results.");
}

// Uncomment to run with the command npx ts-node evals/final-answer.ts
main();