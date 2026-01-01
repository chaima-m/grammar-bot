import google.generativeai as genai
import chromadb
from chromadb.utils import embedding_functions
import os
import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# --- CONFIGURATION ---
# REPLACE THIS WITH YOUR NEW KEY!
API_KEY = "AIzaSyDusmaRBFicQZ8cv20jnBntWHTdlGbB7Fo" 
MODEL_NAME = "gemma-3-27b-it"
DB_PATH = "./grammar_db_flash"

# --- 1. SETUP THE APP ---
app = FastAPI(title="Arabic Grammar RAG API")

# --- 2. DEFINE REQUEST BODY ---
class UserQuery(BaseModel):
    question: str

# --- 3. INITIALIZE BOT (Global Variable) ---
class ArabicGrammarRAG:
    def __init__(self):
        print("Initializing Server...")
        if not API_KEY:
            raise ValueError("API Key is missing!")
        genai.configure(api_key=API_KEY)
        self.model = genai.GenerativeModel(MODEL_NAME)
        
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name="arabic_knowledge_base",
            embedding_function=self.embedding_func
        )
        print("Server Ready!")

    def get_answer(self, query):
        # Retrieve Context
        results = self.collection.query(query_texts=[query], n_results=3)
        context_parts = []
        sources = []
        
        if results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                meta = results['metadatas'][0][i]
                context_parts.append(f"- {doc}")
                sources.append(f"{meta.get('source', 'Unknown')} #{meta.get('ref', '?')}")
        
        context_str = "\n".join(context_parts)
        if not context_str:
            return "لا توجد معلومات كافية.", []

        # IMPROVED PROMPT
        prompt = (
            "لخص الإجابة بدقة نحوية. "
            "أعطِ التعريف المباشر + مثال واحد قصير جداً. "
            "تذكر: الفاعل يأتي غالباً بعد الفعل."
            "لا تشرح ولا تضف تفاصيل غير ضرورية."
            "\n\n"
            f"النص المرجعي: {context_str}\n"
            f"السؤال: {query}\n"
            "الإجابة المختصرة:"
        )

        # Retry Logic
        for attempt in range(3):
            try:
                response = self.model.generate_content(prompt)
                return response.text, sources
            except Exception as e:
                if "429" in str(e):
                    time.sleep(5) # Short wait
                    continue
                return f"Error: {str(e)}", []
        return "Server busy (429). Try again later.", []

# Initialize once on startup
bot = ArabicGrammarRAG()

# --- 4. CREATE THE ENDPOINT ---
@app.post("/ask")
async def ask_endpoint(query: UserQuery):
    if not query.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    answer, sources = bot.get_answer(query.question)
    
    return {
        "answer": answer,
        "sources": sources
    }

# --- 5. RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)