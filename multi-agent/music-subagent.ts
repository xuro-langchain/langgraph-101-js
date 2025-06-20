import { llm, getEngineForChinookDb, queryChinookDb } from './utils';
import { Document } from '@langchain/core/documents';
import { StateGraph, Annotation } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { ToolNode } from '@langchain/langgraph/prebuilt';
import { z } from 'zod';

/* READ
This agent is specialized for helping customers discover and learn about music in the digital catalog.
It provides tools to query music data, including albums, tracks, genres, and song information.
*/

// Initialize database connection
let db: any;

async function initializeDb() {
  if (!db) {
    db = await getEngineForChinookDb();
  }
  return db;
}

/* State ------------------------------------------------------------ 
State is the first LangGraph concept we'll cover.
 **State can be thought of as the memory of the agent - its a shared data structure 
 that's passed on between the nodes of your graph**, 
 representing the current snapshot of your application. 

For this our customer support agent our state will track the following elements: 
1. The customer ID
2. Conversation history
3. Memory from long term memory store
4. Remaining steps, which tracks # steps until it hits recursion limit
*/
const MusicAgentState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: (currentState, updateValue) => [...(currentState || []), ...updateValue],
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

type GraphState = typeof MusicAgentState.State;

/* Tools ------------------------------------------------------------ 
Let's define a list of **tools** our agent will have access to. 
Tools are functionts that can act as extension of the LLM's capabilities. 
In our case, we will create several tools that interacts with the Chinook database regarding music. 

We can create tools using the tool wrapper.
*/
// Tool: Get albums by artist
const getAlbumsByArtist = tool(async (input: { artist: string }) => {
  const dbInstance = await initializeDb();
  const query = `
    SELECT Album.Title, Artist.Name 
    FROM Album 
    JOIN Artist ON Album.ArtistId = Artist.ArtistId 
    WHERE Artist.Name LIKE '%${input.artist}%';
  `;
  const result = await queryChinookDb(dbInstance, query);
  return JSON.stringify(result);
}, {
  name: "get_albums_by_artist",
  description: "Get albums by an artist.",
  schema: z.object({
    artist: z.string()
  })
});

// Tool: Get tracks by artist
const getTracksByArtist = tool(async (input: { artist: string }) => {
  const dbInstance = await initializeDb();
  const query = `
    SELECT Track.Name as SongName, Artist.Name as ArtistName 
    FROM Album 
    LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
    LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
    WHERE Artist.Name LIKE '%${input.artist}%';
  `;
  const result = await queryChinookDb(dbInstance, query);
  return JSON.stringify(result);
}, {
  name: "get_tracks_by_artist",
  description: "Get songs by an artist (or similar artists).",
  schema: z.object({
    artist: z.string()
  })
});

// Tool: Get songs by genre
const getSongsByGenre = tool(async (input: { genre: string }) => {
  const dbInstance = await initializeDb();
  
  // First get genre IDs
  const genreIdQuery = `SELECT GenreId FROM Genre WHERE Name LIKE '%${input.genre}%'`;
  const genreIds = await queryChinookDb(dbInstance, genreIdQuery);
  
  if (!genreIds || genreIds.length === 0) {
    return `No songs found for the genre: ${input.genre}`;
  }
  
  const genreIdList = genreIds.map((gid: any) => gid.GenreId).join(", ");

  // Then get songs for those genres
  const songsQuery = `
    SELECT Track.Name as SongName, Artist.Name as ArtistName
    FROM Track
    LEFT JOIN Album ON Track.AlbumId = Album.AlbumId
    LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId
    WHERE Track.GenreId IN (${genreIdList})
    GROUP BY Artist.Name
    LIMIT 8;
  `;
  const songs = await queryChinookDb(dbInstance, songsQuery);
  
  if (!songs || songs.length === 0) {
    return `No songs found for the genre: ${input.genre}`;
  }
  
  const formattedSongs = songs.map((song: any) => ({
    Song: song.SongName,
    Artist: song.ArtistName
  }));
  
  return JSON.stringify(formattedSongs);
}, {
  name: "get_songs_by_genre",
  description: "Fetch songs from the database that match a specific genre.",
  schema: z.object({
    genre: z.string()
  })
});

// Tool: Check for songs
const checkForSongs = tool(async (input: { song_title: string }) => {
  const dbInstance = await initializeDb();
  const query = `
    SELECT * FROM Track WHERE Name LIKE '%${input.song_title}%';
  `;
  const result = await queryChinookDb(dbInstance, query);
  return JSON.stringify(result);
}, {
  name: "check_for_songs",
  description: "Check if a song exists by its name.",
  schema: z.object({
    song_title: z.string()
  })
});

const musicTools = [
  getAlbumsByArtist, 
  getTracksByArtist, 
  getSongsByGenre, 
  checkForSongs
];

// Bind tools to LLM
// This allows the LLM to know about the tools we've created, and know how to call them.
const llmWithMusicTools = llm.bindTools(musicTools);

/* Nodes ------------------------------------------------------------ 
Now that we have a list of tools, we are ready to build nodes that interact with them. 

Nodes are just JS/TS functions. Nodes take in your graph's State as input, 
execute some logic, and return a new State. 

Here, we're just going to set up 2 nodes for our ReAct agent:
1. **music_assistant**: Reasoning node that decides which function to invoke 
2. **music_tools**: Node that contains all the available tools and executes the function

LangGraph has a pre-built ToolNode that we can utilize to create a node for our tools. 
*/

// Create tool node - LangGraph prebuilt to execute tool calls your LLM can now make
const musicToolNode = new ToolNode(musicTools);

// Music assistant node
async function musicAssistant(state: GraphState): Promise<Partial<GraphState>> {
  // Fetching long term memory
  const memory = state.loaded_memory || "None";

  // Instructions for our agent
  const musicAssistantPrompt = `
  You are a member of the assistant team, your role specifically is to focused on helping customers discover and learn about music in our digital catalog. 
  If you are unable to find playlists, songs, or albums associated with an artist, it is okay. 
  Just inform the customer that the catalog does not have any playlists, songs, or albums associated with that artist.
  You also have context on any saved user preferences, helping you to tailor your response. 
  
  CORE RESPONSIBILITIES:
  - Search and provide accurate information about songs, albums, artists, and playlists
  - Offer relevant recommendations based on customer interests
  - Handle music-related queries with attention to detail
  - Help customers discover new music they might enjoy
  - You are routed only when there are questions related to music catalog; ignore other questions. 
  
  SEARCH GUIDELINES:
  1. Always perform thorough searches before concluding something is unavailable
  2. If exact matches aren't found, try:
     - Checking for alternative spellings
     - Looking for similar artist names
     - Searching by partial matches
     - Checking different versions/remixes
  3. When providing song lists:
     - Include the artist name with each song
     - Mention the album when relevant
     - Note if it's part of any playlists
     - Indicate if there are multiple versions
  
  Additional context is provided below: 

  Prior saved user preferences: ${memory}
  
  Message history is also attached.  
  `;

  // Invoke the model
  const response = await llmWithMusicTools.invoke([
    new SystemMessage(musicAssistantPrompt), 
    ...state.messages
  ]);
  
  // Update the state
  return { messages: [response] };
}

/* Edges ------------------------------------------------------------ 
Now, we need to define a control flow that connects between our defined nodes,
 and that's where the concept of edges come in.

Edges are connections between nodes. They define the flow of the graph.
* **Normal edges** are deterministic and always go from one node to its defined target
* **Conditional edges** are used to dynamically route between nodes, implemented as functions that return the next node to visit based upon some logic. 

In this case, we want a **conditional edge** from our subagent that determines whether to: 
- Invoke tools, or,
- Route to the end if user query has been finished 
*/

// Conditional edge that determines whether to continue or not
function shouldContinue(state: GraphState): string {
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1];
  
  // If there is no function call, then we finish
  if (lastMessage instanceof AIMessage && lastMessage.tool_calls && lastMessage.tool_calls.length > 0) {
    return "continue";
  }
  // Otherwise we finish
  else {
    return "end";
  }
}

// Build the graph
const musicWorkflow = new StateGraph(MusicAgentState);

// Add nodes
musicWorkflow
.addNode("music_assistant", musicAssistant)
.addNode("music_tool_node", musicToolNode)
.addEdge("__start__", "music_assistant")
.addConditionalEdges(
  "music_assistant",
  shouldContinue,
  {
    "continue": "music_tool_node",
    "end": "__end__",
  }
)
.addEdge("music_tool_node", "music_assistant");

export const graph = musicWorkflow.compile({ name: "music_catalog_subagent" });

// Example usage
async function main() {
  const question = "What albums does Queen have?";
  const stream = await graph.stream({ messages: [new HumanMessage(question)] });
  for await (const chunk of stream) {
    const nodeName = Object.keys(chunk)[0];
    const nodeData = chunk[nodeName as keyof typeof chunk];
    
    if (nodeData && typeof nodeData === 'object' && 'messages' in nodeData && Array.isArray(nodeData.messages)) {
      const lastMessage = nodeData.messages[nodeData.messages.length - 1];
      console.log(`\n--- ${nodeName} ---`);
      console.log(`Content: ${lastMessage.content}`);
      
      if ('tool_calls' in lastMessage && Array.isArray(lastMessage.tool_calls) && lastMessage.tool_calls.length > 0) {
        console.log(`Tool calls: ${lastMessage.tool_calls.map((tc: any) => tc.name).join(', ')}`);
      }
    }
  }
}

// Uncomment to run, use command: npx tsx multi-agent/music-subagent.ts
// main(); 