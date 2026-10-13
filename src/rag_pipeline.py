import os
from dotenv import load_dotenv
from langchain.vectorstores import FAISS
from openai import AzureOpenAI
from openai import OpenAI

from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_openai import AzureOpenAIEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain_openai import ChatOpenAI

import time

load_dotenv()

azure_embeddings_api_key = os.getenv("azure_embeddings_api_key")
azure_embeddings_endpoint = os.getenv("azure_embeddings_endpoint")
azure_embeddings_deployment = os.getenv("azure_embeddings_deployment")

azure_api_key = os.getenv("azure_api_key")
azure_endpoint = os.getenv("azure_endpoint")
azure_deployment = os.getenv("azure_deployment")


# Creating RAG Pipeline

def load_pipeline():
    #Create OPENAI Embeddings

    embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_VERSION"),
    azure_deployment="text-embedding-ada-002"
)

    # Create Vector Database
    faiss_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    llm = ChatOpenAI(
    base_url=os.getenv("AZURE_OPENAI_CHAT_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_CHAT_API_KEY"),
    model="gpt-5-mini"
)

    # Add Memory to the RAG Pipeline
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

    # Semantic Retriver
    semantic_retriever = faiss_store.as_retriever(search_type="mmr", search_kwargs={"k": 8})

    # Create Document from Vector Store

    documents = list(faiss_store.docstore._dict.values())

    # Create BM25 Retriever
    bm25_retriever = BM25Retriever.from_documents(documents=documents, k=8)

    from langchain.retrievers import EnsembleRetriever
    ensemble_retriever = EnsembleRetriever(retrievers=[semantic_retriever, bm25_retriever], weights=[0.6, 0.4])

    reranker = CohereRerank(
        cohere_api_key=os.getenv("COHERE_API_KEY"),
        model = "rerank-v3.5",
        top_n=4)

    contexual_retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=ensemble_retriever
    )

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template="""You are a Healthcare Knowledge Assistant that provides
evidence-based health information using Retrieval-Augmented Generation (RAG).

Your task is to answer the user's question using ONLY the
provided context.

CONTEXT:
{context}

QUESTION:
{question}

Chat History:
{chat_history}


INSTRUCTIONS:
1. Answer using the information available in the context.
2. Explain relevant symptoms, possible causes, diagnostic methods,
   treatments, and prevention when the context supports them.
3. Do not invent medical facts or assume missing information.
4. If the context does not contain enough information, clearly state:
   "The available medical sources do not provide enough information
   to answer this question reliably."
5. Do not provide a definitive diagnosis or prescribe medication.
6. If discussing possible conditions, explain that they are possibilities
   and that a qualified healthcare professional must assess the patient.
7. Provide citations using the source metadata available in the context.
   Do not fabricate sources, titles, or references.
8. Use clear, simple, professional language.
9. Highlight urgent warning signs when supported by the retrieved context.

FORMAT YOUR RESPONSE AS:

Answer:
[Direct answer to the user's question]

Relevant Medical Information:
[Symptoms, causes, diagnostic methods, treatment, or prevention,
where supported by the context]

Sources:
healthcare_disease_dataset.csv,
medical_dataset.csv,
Symptom2Disease.csv

Medical Disclaimer:
This information is for educational purposes only and is not a
substitute for professional medical advice, diagnosis, or treatment.
"""
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=contexual_retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    return chain

def ask_question(chain, question):
    start = time.time()
    result = chain.invoke({"question": question})
    latency = time.time() - start
    docs = result['source_documents']

    retrieved_docs = [doc.page_content for doc in docs]
    sources = [doc.metadata for doc in docs]

    return {
        "answer": result['answer'],
        "retrieved_docs": retrieved_docs,
        "sources": sources,
        "latency": latency
    }