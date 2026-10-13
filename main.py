from fastapi import FastAPI
from pydantic import BaseModel
from src.agent import healthcare_graph

app = FastAPI(
    title = "HealthCare Knowledge Assistant",
    version="1.0",
    description="An agentic RAG application that retrieves medical "
                "information and generates context-based responses."
)

class QueryRequest(BaseModel):
    question: str
    thread_id: str = "default"

@app.get("/")
def home():
    return {"message": "Healthcare Knowledge Assistant is running"}

@app.post("/query")
def ask_question(request: QueryRequest):
    result = healthcare_graph.invoke(
        {
            "question": request.question,
            "context": "",
            "sources": [],
            "retriever_output": "",
            "consultation_response": "",
            "possible_conditions": [],
            "final_response": ""
        },
        config={"configurable": {"thread_id": request.thread_id}}
    )

    return {
        "question": request.question,
        "answer": result.get("final_response", ""),
        "sources": result.get("sources", [])
    }
 