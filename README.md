Capstone-2 Agentic RAG/
│
├── .github/
│   └── workflow/
│       └── ci.yml
│
├── data/
│   ├── healthcare_disease_dataset.csv
│   ├── medical_dataset.csv
│   └── Symptom2Disease.csv
│
├── faiss_index/
│   ├── index.faiss
│   └── index.pkl
│
├── src/
│   ├── __pycache__/
│   ├── agent.py                # Agent Orchestration
│   ├── demo.ipynb              # Development & Testing
│   ├── ingest.py               # Data Ingestion & Embedding Creation
│   └── rag_pipeline.py         # Retrieval-Augmented Generation Pipeline
│
├── .env                        # API Keys & Environment Variables
├── .gitignore
├── app.py                      # Streamlit Frontend
├── main.py                     # Application Entry Point
├── README.md
└── requirements.txt