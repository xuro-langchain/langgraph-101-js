import * as fs from 'fs';
import * as path from 'path';
import axios from 'axios';
import sqlite3 from 'sqlite3';
import { ChatOpenAI } from '@langchain/openai';
import { OpenAIEmbeddings } from '@langchain/openai';
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { Document } from "@langchain/core/documents"; // Optional: if you want to split Document objects
import { PuppeteerWebBaseLoader } from "@langchain/community/document_loaders/web/puppeteer";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

// NOTE: Configure the LLM that you want to use
export const llm = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0,
});

// Alternative models (uncomment to use):
// import { ChatAnthropic } from '@langchain/anthropic';
// import { ChatVertexAI } from '@langchain/google-vertexai';
// export const llm = new ChatAnthropic({
//   modelName: 'claude-3-5-sonnet-20240620',
//   temperature: 0,
// });
// export const llm = new ChatVertexAI({
//   modelName: 'gemini-1.5-flash-002',
//   temperature: 0,
// });

// NOTE: Configure the embedding model that you want to use
export const embeddingModel = new OpenAIEmbeddings();

const LANGGRAPH_DOCS = [
  'https://langchain-ai.github.io/langgraph/',
  'https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/',
  'https://langchain-ai.github.io/langgraph/tutorials/chatbots/information-gather-prompting/',
  'https://langchain-ai.github.io/langgraph/tutorials/code_assistant/langgraph_code_assistant/',
  'https://langchain-ai.github.io/langgraph/tutorials/multi_agent/multi-agent-collaboration/',
  'https://langchain-ai.github.io/langgraph/tutorials/multi_agent/agent_supervisor/',
  'https://langchain-ai.github.io/langgraph/tutorials/multi_agent/hierarchical_agent_teams/',
  'https://langchain-ai.github.io/langgraph/tutorials/plan-and-execute/plan-and-execute/',
  'https://langchain-ai.github.io/langgraph/tutorials/rewoo/rewoo/',
  'https://langchain-ai.github.io/langgraph/tutorials/llm-compiler/LLMCompiler/',
  'https://langchain-ai.github.io/langgraph/concepts/high_level/',
  'https://langchain-ai.github.io/langgraph/concepts/low_level/',
  'https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/',
  'https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/',
  'https://langchain-ai.github.io/langgraph/concepts/multi_agent/',
  'https://langchain-ai.github.io/langgraph/concepts/persistence/',
  'https://langchain-ai.github.io/langgraph/concepts/streaming/',
  'https://langchain-ai.github.io/langgraph/concepts/faq/',
];

export async function getLanggraphDocsRetriever() {
  console.log('Starting document loading...');
  
  // Load the documents and create in-memory vectorstore
  const docsPromises = LANGGRAPH_DOCS.map(async (url, index) => {
    console.log(`Loading document ${index + 1}/${LANGGRAPH_DOCS.length}: ${url}`);
    try {
      const docs = await new PuppeteerWebBaseLoader(url).load();
      console.log(`Successfully loaded ${docs.length} documents from ${url}`);
      return docs;
    } catch (error) {
      console.error(`Failed to load ${url}:`, error);
      return [];
    }
  });
  
  const docsArrays = await Promise.all(docsPromises);
  const docsList = docsArrays.flat();
  
  console.log(`Total documents loaded: ${docsList.length}`);

  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 200,
    chunkOverlap: 0,
  });

  const docSplits = await textSplitter.splitDocuments(docsList);
  console.log(`Documents split into ${docSplits.length} chunks`);
  
  // Use in-memory vectorstore
  const vectorstore = await MemoryVectorStore.fromDocuments(
    docSplits,
    embeddingModel
  );

  console.log('In-memory vectorstore created');
  return vectorstore.asRetriever({ k: 4 });
}