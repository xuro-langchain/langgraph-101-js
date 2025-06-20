import { getLanggraphDocsRetriever, llm } from './utils';
import { HumanMessage } from '@langchain/core/messages';
import { Document } from '@langchain/core/documents';
import { StateGraph, START, END, Annotation } from '@langchain/langgraph';

// Initialize retriever in a function to avoid top-level await
let retriever: any;

async function initializeRetriever() {
  if (!retriever) {
    retriever = await getLanggraphDocsRetriever();
  }
  return retriever;
}

/* Define the State using Annotation
State is one of the most important concepts in an Agent. 
When defining a Graph, you must pass in a schema for State. 
The State schema serves as the input schema for all Nodes and Edges in the graph. 
Let's use the Annotation class, which provides type hints for the properties of the state.

The State of our RAG application will keep track of the user's question, our RAG app's LLM generated response, 
and the list of retrieved relevant documents.
*/
const SimpleRAGState = Annotation.Root({
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
});
type GraphState = typeof SimpleRAGState.State;

/* Define the RAG prompt
We define the prompt we will use in our RAG application. This will provide instructions to the 
LLM we use in our application.
*/
const RAG_PROMPT = `You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.

Question: {question} 
Context: {context} 
Answer:`;

/* Define the Nodes
Nodes are just typescript functions. As mentioned above, Nodes take in your graph's State as input. 

The first positional argument is the state, as defined above.
Each node can access each property in the state, ex. `state.question`.
  
Nodes return any updates to the state that they want to make. 
By default, the new value returned by each node will override the prior state value. 
You can implement custom handling for updates to State using State Reducers, which we see in our Annotations.

Here, we're going to set up two nodes for our RAG flow:
1. retrieveDocuments: Retrieves documents from our vector store
2. generateResponse: Generates an answer from our documents
*/
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
  const formattedDocs = documents.map(doc => doc.pageContent).join('\n\n');
  
  // Invoke our LLM with our RAG prompt
  const ragPromptFormatted = RAG_PROMPT
    .replace('{context}', formattedDocs)
    .replace('{question}', question);
  
  const generation = await llm.invoke([new HumanMessage(ragPromptFormatted)]);
  return { generation: generation.content as string };
}

/* Defining Edges
Edges define how your agentic applications progresses from each Node to the next Node.
- Normal Edges are used if you want to *always* go from, for example, `node_1` to `node_2`.
- Conditional Edges are used want to *optionally* route between nodes.
 
Conditional edges are implemented as functions that return the next node to visit based upon some logic.
Note that these functions often use values from our graph's State to determine how to traverse.

For our simple RAG application, all we need are normal edges.
*/
const graphBuilder = new StateGraph(SimpleRAGState);

// Build the graph
graphBuilder
.addNode('retrieve_documents', retrieveDocuments)
.addNode('generate_response', generateResponse)
.addEdge(START, 'retrieve_documents')
.addEdge('retrieve_documents', 'generate_response')
.addEdge('generate_response', END)

// Compile the graph
export const graph = graphBuilder.compile(); 

// Now we can run the graph!
async function main() {
  const question = "Does LangGraph work with OSS LLMs?";
  const result = await graph.invoke({ question });
  console.log("Documents --------------------------------")
  console.log(result["documents"]);
  console.log("Generation --------------------------------")
  console.log(result["generation"]);
}

// npx ts-node rag-agents/simple-rag.ts
// Uncomment this to run the graph with the above command
main();