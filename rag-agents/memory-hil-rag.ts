import { getLanggraphDocsRetriever, llm } from './utils';
import { Document } from '@langchain/core/documents';
import { StateGraph, Annotation, Command } from '@langchain/langgraph';
import { interrupt } from "@langchain/langgraph";
import { getBufferString } from "@langchain/core/messages";
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { MemorySaver } from '@langchain/langgraph';

/* READ
In every example so far, state has been transient to a single graph execution. 
If we invoke our graph for a second time, we are starting with a fresh state. 
This limits our ability to have multi-turn conversations with interruptions. 
We can use persistence to address this! 
 
LangGraph can use a checkpointer to automatically save the graph state after each step. 
This built-in persistence layer gives us memory, allowing LangGraph to pick up from the last state update.

We'll also show in this section how to incorporate human-in-the-loop into our application.
Sometimes you need a human to provide oversight to your application, which LangGraph makes possible 
through interrupts
*/

// Initialize retriever in a function to avoid top-level await
let retriever: any;

async function initializeRetriever() {
  if (!retriever) {
    retriever = await getLanggraphDocsRetriever();
  }
  return retriever;
}

/* 
Before we set up memory in our application, let's edit our State and Nodes so that instead of 
acting a single "question", we instead act on a list of "questions and answers".

We'll call our list "messages". These existing messages will all be used for our retrieval step. 
And at the end of our flow when our LLM responds, we will add the latest question 
and answer to our "messages" history. 
*/
const MemoryHILRAGState = Annotation.Root({
  question: Annotation<string>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => '',
  }),
  messages: Annotation<BaseMessage[]>({
    reducer: (currentState, updateValue) => [...(currentState || []), ...updateValue],
    default: () => [],
  }),
  generation: Annotation<string>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => '',
  }),
  documents: Annotation<Document[]>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => [],
  }),
  attempted_generations: Annotation<number>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => 0,
  }),
});

type GraphState = typeof MemoryHILRAGState.State;

// -----------------------------------------------------------------------------------
// NEW: Update Prompts to Provide Chat History ---------------------------------------
// -----------------------------------------------------------------------------------

/*
Now let's edit our existing Nodes to use `messages` in addition to `question`, 
specifically for grading document relevance, and generating a response.
*/
const RAG_PROMPT_WITH_CHAT_HISTORY = `You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the latest question in the conversation. 
If you don't know the answer, just say that you don't know. 
The pre-existing conversation may provide important context to the question.
Use three sentences maximum and keep the answer concise.

Existing Conversation:
{conversation}

Latest Question:
{question}

Additional Context from Documents:
{context} 

Answer:`;

async function retrieveDocuments(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---RETRIEVE DOCUMENTS---');
  const question = state.question;
  const retrieverInstance = await initializeRetriever();
  const documents = await retrieverInstance.invoke(question);
  return { documents };
}

// -----------------------------------------------------------------------------------
// NEW: Add Interuppt to Allow Human-in-the-Loop  ------------------------------------
// -----------------------------------------------------------------------------------

/*
When building human-in-the-loop into Python programs, one common way to do this is with the input function. 
With this, your program pauses, a text box pops up in your terminal, and whatever you type 
is then used as the response to that function. You use it like the below:

`response = input("Your question here")`

We've tried to emulate this developer experience by adding a new function to LangGraph: interrupt. 
You can use this in much the same way as input:

`response = interrupt("Your question here")`

This is designed to work in production settings. When you do this, it will pause execution of the graph, 
mark the thread you are running as interrupted, and put whatever you passed as an input to 
interrupt into the persistence layer. This way, you can check the thread status, 
see that it's interrupted, check the message, and then based on that invoke the graph again 
(in a special way) to pass your response back in:

`graph.invoke(Command(resume="Your response here"), thread)`

Note that it doesn't function exactly the same as input 
(it reruns any work in that node done before this is called, but no previous nodes). 
This ensures interrupted threads don't take up any resources (beyond storage space), 
and can be resumed many months later, on a different machine, etc.
*/
async function generateResponse(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---GENERATE RESPONSE---');
  
  // Use the interrupt mechanism to ask for additional context
  const additional_context = await interrupt("Do you have anything else to add that you think is relevant?");
  
  const question = state.question;
  const documents = state.documents;
  const messages = state.messages || [];
  const attemptedGenerations = state.attempted_generations || 0;
  
  // For simplicity, we'll just append the additional context to the conversation history
  const conversation = getBufferString(messages) + additional_context;
  const formattedDocs = documents.map(doc => doc.pageContent).join('\n\n');
  
  const ragPromptFormatted = RAG_PROMPT_WITH_CHAT_HISTORY
    .replace('{context}', formattedDocs)
    .replace('{conversation}', conversation)
    .replace('{question}', question);
  
  const generation = await llm.invoke([new HumanMessage(ragPromptFormatted)]);
  return {
    generation: generation.content as string,
    attempted_generations: attemptedGenerations + 1,
  };
}

// -----------------------------------------------------------------------------------
// Document Grading ------------------------------------------------------------
// -----------------------------------------------------------------------------------

// Define the schema for grading documents
const GradeDocumentsSchema = {
  type: "object",
  properties: {
    is_relevant: {
      type: "boolean",
      description: "The document is relevant to the question, true or false"
    }
  },
  required: ["is_relevant"]
};

const gradeDocumentsLlm = llm.withStructuredOutput(GradeDocumentsSchema);
const gradeDocumentsSystemPrompt = `You are a grader assessing relevance of a retrieved document to a conversation between a user and an AI assistant, and user's latest question. 
If the document contains keyword(s) or semantic meaning related to the user question, definitely grade it as relevant. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals that are not relevant at all. 
Give a binary score true or false to indicate whether the document is relevant to the question.`;
const gradeDocumentsPrompt = 'Here is the retrieved document: \n\n {document} \n\n Here is the conversation so far: \n\n {conversation} \n\n Here is the user question: \n\n {question}';

async function gradeDocuments(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---CHECK DOCUMENT RELEVANCE TO QUESTION---');
  const question = state.question;
  const documents = state.documents;
  const messages = state.messages || [];
  const conversation = getBufferString(messages);
  
  // Score each doc
  const filteredDocs: Document[] = [];
  for (const doc of documents) {
    const gradeDocumentsPromptFormatted = gradeDocumentsPrompt
      .replace('{document}', doc.pageContent)
      .replace('{question}', question)
      .replace('{conversation}', conversation);
    
    const score = await gradeDocumentsLlm.invoke([
      new SystemMessage(gradeDocumentsSystemPrompt),
      new HumanMessage(gradeDocumentsPromptFormatted)
    ]);
    
    const grade = score.is_relevant;
    if (grade) {
      console.log('---GRADE: DOCUMENT RELEVANT---');
      filteredDocs.push(doc);
    } else {
      console.log('---GRADE: DOCUMENT NOT RELEVANT---');
      continue;
    }
  }
  return { documents: filteredDocs };
}

function decideToGenerate(state: GraphState): string {
  console.log('---ASSESS GRADED DOCUMENTS---');
  const filteredDocuments = state.documents;

  if (!filteredDocuments.length) {
    console.log('---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, END---');
    return 'none relevant';
  } else {
    // We have relevant documents, so generate answer
    console.log('---DECISION: GENERATE---');
    return 'some relevant';
  }
}

// -----------------------------------------------------------------------------------
// Hallucination Checking -------------------------------------------------------
// -----------------------------------------------------------------------------------

// Define the schema for grading hallucinations
const GradeHallucinationsSchema = {
  type: "object",
  properties: {
    grounded_in_facts: {
      type: "boolean",
      description: "Answer is grounded in the facts, true or false"
    }
  },
  required: ["grounded_in_facts"]
};

const gradeHallucinationsLlm = llm.withStructuredOutput(GradeHallucinationsSchema);
const gradeHallucinationsSystemPrompt = `You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. 
Give a binary score true or false. True means that the answer is grounded in / supported by the set of facts.`;
const gradeHallucinationsPrompt = 'Set of facts: \n\n {documents} \n\n LLM generation: {generation}';

const ATTEMPTED_GENERATION_MAX = 3;

async function gradeHallucinations(state: GraphState): Promise<string> {
  console.log('---CHECK HALLUCINATIONS---');
  const documents = state.documents;
  const generation = state.generation;
  const attemptedGenerations = state.attempted_generations;

  const formattedDocs = documents.map(doc => doc.pageContent).join('\n\n');

  const gradeHallucinationsPromptFormatted = gradeHallucinationsPrompt
    .replace('{documents}', formattedDocs)
    .replace('{generation}', generation);

  const score = await gradeHallucinationsLlm.invoke([
    new SystemMessage(gradeHallucinationsSystemPrompt),
    new HumanMessage(gradeHallucinationsPromptFormatted)
  ]);
  
  const grade = score.grounded_in_facts;

  // Check hallucination
  if (grade) {
    console.log('---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---');
    return 'supported';
  } else if (attemptedGenerations >= ATTEMPTED_GENERATION_MAX) {
    console.log('---DECISION: TOO MANY ATTEMPTS, GIVE UP---');
    throw new Error('Too many attempted generations with hallucinations, giving up.');
  } else {
    console.log('---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---');
    return 'not supported';
  }
}

// -----------------------------------------------------------------------------------
// New: Memory Configuration ---------------------------------------------------------
// -----------------------------------------------------------------------------------

async function configureMemory(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---CONFIGURE MEMORY---');
  const question = state.question;
  const generation = state.generation;
  
  // Add the question and generation to our message history
  const newMessages = [
    new HumanMessage(question),
    new AIMessage(generation)
  ];
  
  return {
    messages: newMessages,
    attempted_generations: 0,   // Reset this value to 0
    documents: []    // Reset documents to empty
  };
}

// Build the graph
const graphBuilder = new StateGraph(MemoryHILRAGState);

graphBuilder
.addNode('retrieve_documents', retrieveDocuments)
.addNode('generate_response', generateResponse)
.addNode('grade_documents', gradeDocuments)
.addNode('configure_memory', configureMemory)
.addEdge("__start__", 'retrieve_documents')
.addEdge('retrieve_documents', 'grade_documents')
.addConditionalEdges(
  'grade_documents',
  decideToGenerate,
  {
    'some relevant': 'generate_response', // generate response if at least one document is relevant
    'none relevant': "__end__", // end execution if no documents are relevant
  }
)
.addConditionalEdges(
  'generate_response',
  gradeHallucinations,
  {
    'supported': 'configure_memory', // configure memory if hallucination is not detected
    'not supported': 'generate_response', // re-generate response if hallucinated
  }
)
.addEdge('configure_memory', "__end__");

// New: Compile the Graph with Memory -----------------------------------------------
/* 
Let's define our graph and add some local memory! One of the easiest to work with is `MemorySaver`, 
an in-memory key-value store for Graph state.

All we need to do is compile the graph with a checkpointer, and our graph has memory!
*/
const memory = new MemorySaver();
export const graph = graphBuilder.compile({ checkpointer: memory });

// -----------------------------------------------------------------------------------
// New: Run the Graph with Memory ---------------------------------------------------
// -----------------------------------------------------------------------------------

/*
When we use memory, we need to specify a `thread_id`.

This `thread_id` will store our collection of graph states.

* The checkpointer write the state at every step of the graph
* These checkpoints are saved in a thread 
* We can access that thread in the future using the `thread_id`
*/
async function main() {
    const thread_id = crypto.randomUUID();
    const config = { configurable: { thread_id } };
    const question = "Can I use LangGraph for customer support? I want to create an agent application.";
    const first_response = await graph.invoke({ question }, config);
    for (const m of first_response["messages"]) {
        console.log(`${m.constructor.name}: ${m.content}`);
    }
    console.log("--------------------------------")
    console.log("Graph has run until interrupt! Let's check out the state")
    const state = await graph.getState(config);
    console.log(state.values)
    console.log("Next node: ", state.next)
    console.log("--------------------------------")
    console.log("Let's resume the graph with a user response")
    const command = new Command({resume: "I am building an airline booking agent. Please mention this in your response."});
    const second_response = await graph.invoke(command, config);
    for (const m of second_response["messages"]) {
        console.log(`${m.constructor.name}: ${m.content}`);
    }
    console.log("--------------------------------")
    console.log("Our added context about making an airline booking agent is correctly mentioned in the response")
}

// npx ts-node rag-agents/memory-hil-rag.ts
// Uncomment this to run the graph with the above command
main(); 