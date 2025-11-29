from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
os.environ["GROQ_API_KEY"] = "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"
Settings.llm = Groq(model="llama-3.3-70b-versatile")
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")


documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

print("---ready to use ---")


app = FastAPI()


class Question(BaseModel):
    text: str
@app.post("/ask")
async def ask_question(question: Question):
    try:
        response = query_engine.query(question.text)
        return {"answer": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/")
def read_root():
    return {"message": "Hello! The RAG API is running."}