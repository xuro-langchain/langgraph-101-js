import { llm, getEngineForChinookDb, queryChinookDb } from './utils';
import { StateGraph, Annotation } from '@langchain/langgraph';
import { HumanMessage, SystemMessage, AIMessage, BaseMessage } from '@langchain/core/messages';
import { tool } from '@langchain/core/tools';
import { createReactAgent } from '@langchain/langgraph/prebuilt';
import { z } from 'zod';

/* READ
LangGraph offers pre-built libraries for common architectures, 
allowing us to quickly create architectures like ReAct or multi-agent architacture.
 
In the last workflow, we have seen how we can build a ReAct agent from scratch (music-subagent). 
Now, we will show how we can leverage the LangGraph pre-built libraries to achieve similar results. 

This agent is specialized for retrieving and processing invoice information from a database.
It provides tools to query invoice data, customer information, and employee details.
*/

// Initialize database connection
let db: any;

async function initializeDb() {
  if (!db) {
    db = await getEngineForChinookDb();
  }
  return db;
}

// Define the state using Annotation
const InvoiceAgentState = Annotation.Root({
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

type GraphState = typeof InvoiceAgentState.State;

// Tool: Get invoices by customer sorted by date
const getInvoicesByCustomerSortedByDate = tool(async (input: { customer_id: string }) => {
  const dbInstance = await initializeDb();
  const query = `SELECT * FROM Invoice WHERE CustomerId = ${input.customer_id} ORDER BY InvoiceDate DESC;`;
  const result = await queryChinookDb(dbInstance, query);
  return JSON.stringify(result);
}, {
  name: "get_invoices_by_customer_sorted_by_date",
  description: "Look up all invoices for a customer using their ID. The invoices are sorted in descending order by invoice date, which helps when the customer wants to view their most recent/oldest invoice, or if they want to view invoices within a specific date range.",
  schema: z.object({
    customer_id: z.string()
  })
});

// Tool: Get invoices sorted by unit price
const getInvoicesSortedByUnitPrice = tool(async (input: { customer_id: string }) => {
  const dbInstance = await initializeDb();
  const query = `
    SELECT Invoice.*, InvoiceLine.UnitPrice
    FROM Invoice
    JOIN InvoiceLine ON Invoice.InvoiceId = InvoiceLine.InvoiceId
    WHERE Invoice.CustomerId = ${input.customer_id}
    ORDER BY InvoiceLine.UnitPrice DESC;
  `;
  const result = await queryChinookDb(dbInstance, query);
  return JSON.stringify(result);
}, {
  name: "get_invoices_sorted_by_unit_price",
  description: "Use this tool when the customer wants to know the details of one of their invoices based on the unit price/cost of the invoice. This tool looks up all invoices for a customer, and sorts the unit price from highest to lowest. In order to find the invoice associated with the customer, we need to know the customer ID.",
  schema: z.object({
    customer_id: z.string()
  })
});

// Tool: Get employee by invoice and customer
const getEmployeeByInvoiceAndCustomer = tool(async (input: { invoice_id: string; customer_id: string }) => {
  const dbInstance = await initializeDb();
  const query = `
    SELECT Employee.FirstName, Employee.Title, Employee.Email
    FROM Employee
    JOIN Customer ON Customer.SupportRepId = Employee.EmployeeId
    JOIN Invoice ON Invoice.CustomerId = Customer.CustomerId
    WHERE Invoice.InvoiceId = (${input.invoice_id}) AND Invoice.CustomerId = (${input.customer_id});
  `;
  
  const result = await queryChinookDb(dbInstance, query);
  
  if (!result || result.length === 0) {
    return `No employee found for invoice ID ${input.invoice_id} and customer identifier ${input.customer_id}.`;
  }
  return JSON.stringify(result);
}, {
  name: "get_employee_by_invoice_and_customer",
  description: "This tool will take in an invoice ID and a customer ID and return the employee information associated with the invoice.",
  schema: z.object({
    invoice_id: z.string(),
    customer_id: z.string()
  })
});

const invoiceTools = [
  getInvoicesByCustomerSortedByDate, 
  getInvoicesSortedByUnitPrice, 
  getEmployeeByInvoiceAndCustomer
];

const invoiceSubagentPrompt = `
You are a subagent among a team of assistants. You are specialized for retrieving and processing invoice information. You are routed for invoice-related portion of the questions, so only respond to them.. 

You have access to three tools. These tools enable you to retrieve and process invoice information from the database. Here are the tools:
- get_invoices_by_customer_sorted_by_date: This tool retrieves all invoices for a customer, sorted by invoice date.
- get_invoices_sorted_by_unit_price: This tool retrieves all invoices for a customer, sorted by unit price.
- get_employee_by_invoice_and_customer: This tool retrieves the employee information associated with an invoice and a customer.

If you are unable to retrieve the invoice information, inform the customer you are unable to retrieve the information, and ask if they would like to search for something else.

CORE RESPONSIBILITIES:
- Retrieve and process invoice information from the database
- Provide detailed information about invoices, including customer details, invoice dates, total amounts, employees associated with the invoice, etc. when the customer asks for it.
- Always maintain a professional, friendly, and patient demeanor

You may have additional context that you should use to help answer the customer's query. It will be provided to you below:
`;

// --------------------------------------------------------------------------------
// New: Create the React agent using the prebuilt function ------------------------
// --------------------------------------------------------------------------------
export const graph = createReactAgent({
  llm,
  tools: invoiceTools,
  name: "invoice_information_subagent",
  prompt: invoiceSubagentPrompt,
  stateSchema: InvoiceAgentState,
});

// Example usage
async function main() {
  const question = "What are the invoices for customer ID 1?";
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

// Uncomment to run, use command: npx tsx multi-agent/invoice-subagent.ts
// main(); 