import axios from 'axios';
import * as sqlite3 from 'sqlite3';
import { ChatOpenAI } from '@langchain/openai';

// NOTE: Configure the LLM that you want to use
export const llm = new ChatOpenAI({
  modelName: 'gpt-4o',
  temperature: 0,
});

export function getEngineForChinookDb(): Promise<sqlite3.Database> {
  return new Promise((resolve, reject) => {
    // Create in-memory SQLite database
    const db = new sqlite3.Database(':memory:', (err) => {
      if (err) {
        reject(err);
        return;
      }

      // Load the Chinook database schema
      const chinookUrl = 'https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql';
      
      axios.get(chinookUrl)
        .then((response) => {
          const sqlScript = response.data;
          db.exec(sqlScript, (err) => {
            if (err) {
              console.error('Error loading Chinook database:', err);
              reject(err);
            } else {
              console.log('Chinook database loaded successfully');
              resolve(db);
            }
          });
        })
        .catch((error) => {
          console.error('Error fetching Chinook database schema:', error);
          reject(error);
        });
    });
  });
}

// Helper function to query the Chinook database
export function queryChinookDb(db: sqlite3.Database, query: string): Promise<any[]> {
  return new Promise((resolve, reject) => {
    db.all(query, (err, rows) => {
      if (err) {
        reject(err);
      } else {
        resolve(rows);
      }
    });
  });
} 