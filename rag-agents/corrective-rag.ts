import { getLanggraphDocsRetriever, llm } from './utils';
import { Document } from '@langchain/core/documents';
import { StateGraph, Annotation } from '@langchain/langgraph';
import { HumanMessage, SystemMessage } from '@langchain/core/messages';

/* READ
In this section, we're going to add a few techniques that can improve our RAG workflow. 
Specifically, we'll introduce
- Document Grading: Are the documents fetched by the retriever actually relevant to the user's question?
- Hallucination Checking: Is our generated answer actually grounded in the documents?

We're also going to add some constraints to the inputs and outputs of our application 
for the best user experience.
*/

// Initialize retriever in a function to avoid top-level await
let retriever: any;

async function initializeRetriever() {
  if (!retriever) {
    retriever = await getLanggraphDocsRetriever();
  }
  return retriever;
}

// Define the state using Annotation
const CorrectiveRAGState = Annotation.Root({
  question: Annotation<string>({
    reducer: (currentState, updateValue) => updateValue,
    default: () => '',
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

type GraphState = typeof CorrectiveRAGState.State;

const RAG_PROMPT = `You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:`;

async function retrieveDocuments(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---RETRIEVE DOCUMENTS---');
  const question = state.question;
  const retrieverInstance = await initializeRetriever();
  const documents = await retrieverInstance.invoke(question);
  return { documents };
}

async function generateResponse(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---GENERATE RESPONSE---');
  const question = state.question;
  const documents = state.documents;
  const attemptedGenerations = state.attempted_generations || 0;
  const formattedDocs = documents.map(doc => doc.pageContent).join('\n\n');
  
  // Invoke our LLM with our RAG prompt
  const ragPromptFormatted = RAG_PROMPT
    .replace('{context}', formattedDocs)
    .replace('{question}', question);
  
  const generation = await llm.invoke([new HumanMessage(ragPromptFormatted)]);
  return {
    generation: generation.content as string,
    attempted_generations: attemptedGenerations + 1,
  };
}

// -----------------------------------------------------------------------------------
// NEW: Document Grading ------------------------------------------------------------
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

/*
Some LLMs provide support for Structured Outputs, which provides a typing guarantee for the output schema 
of the LLM's response. Here, we can our defined class to specify return type. 
The provided description helps the LLM generate the value for the field.

We can hook this up to our previously defined `llm` using `withStructuredOutput`. 
Now, when we invoke our `gradeDocumentsLlm`, we can expect the returned object
 to contain the expected field.
*/
const gradeDocumentsLlm = llm.withStructuredOutput(GradeDocumentsSchema);
const gradeDocumentsSystemPrompt = `You are a grader assessing relevance of a retrieved document to a user question. 
If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. 
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. 
Give a binary score true or false to indicate whether the document is relevant to the question.`;
const gradeDocumentsPrompt = 'Here is the retrieved document: \n\n {document} \n\n Here is the user question: \n\n {question}';

async function gradeDocuments(state: GraphState): Promise<Partial<GraphState>> {
  console.log('---GRADE DOCUMENTS---');
  const question = state.question;
  const documents = state.documents;
  
  // Score each doc
  const filteredDocs: Document[] = [];
  for (const doc of documents) {
    const gradeDocumentsPromptFormatted = gradeDocumentsPrompt
      .replace('{document}', doc.pageContent)
      .replace('{question}', question);
    
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

/*
Let's make sure that at least some documents are relevant if we are going to respond to the user! 
To do this, we need to add a conditional edge function. 
Once we define this edge, we'll add it in when constructing the final Graph
*/
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
// NEW: Hallucination Checking -------------------------------------------------------
// -----------------------------------------------------------------------------------

/*
Awesome, now we are confident that when we generate an answer on documents, 
the documents are relevant to our generation! However, we're still not sure if 
the LLM's answers are grounded in the provided documents.

For sensitive use cases (ex. legal, healthcare, etc.), it is really 
important that your LLM application is not hallucinating. 
Let's add an explicit hallucination grader to gain more confidence!
*/

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

/*
Let's add an edge function for grading hallucinations after our LLM generates a response. 
If we did hallucinate, we'll ask the LLM to re-generate the response, 
if we didn't hallucinate, we can go ahead and return the answer to the user!

Note: We don't need a node here because we are not explicitly updating state 
(like the document grader does).
*/
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

// Build the graph
const graphBuilder = new StateGraph(CorrectiveRAGState);

graphBuilder
.addNode('retrieve_documents', retrieveDocuments)
.addNode('generate_response', generateResponse)
.addNode('grade_documents', gradeDocuments)
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
    'supported': "__end__", // end and return answer if hallucination is not detected
    'not supported': 'generate_response', // re-generate response if hallucinated
  }
);

export const graph = graphBuilder.compile(); 

// Now we can run the graph!
async function main() {
    const question = "Does LangGraph help with customer support bots?";
    const result = await graph.invoke({ question });
    console.log("Documents --------------------------------")
    console.log(result["documents"]);
    console.log("--------------------------------")
    console.log(result["generation"]);
  }

// npx ts-node rag-agents/corrective-rag.ts
// Uncomment this to run the graph with the above command
main();