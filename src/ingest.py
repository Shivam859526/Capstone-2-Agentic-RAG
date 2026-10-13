# 1. Import Library
from langchain_community.document_loaders import JSONLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from openai import AzureOpenAI
from openai import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain_openai import AzureOpenAIEmbeddings
import os
 
load_dotenv()

azure_embeddings_api_key = os.getenv("azure_embeddings_api_key")
azure_embeddings_endpoint = os.getenv("azure_embeddings_endpoint")
azure_embeddings_deployment = os.getenv("azure_embeddings_deployment")
azure_api_version = os.getenv("azure_api_version")
#2 Load The DataSet
csv_loader = DirectoryLoader("./data", glob="*.csv", loader_cls=CSVLoader, loader_kwargs={"csv_args": {"delimiter": ","}})

medical_docs = csv_loader.load()


# 3. Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                               chunk_overlap=100,
                                               separators=["\n\n", "\n", " ", ""])

medical_chunks = text_splitter.split_documents(medical_docs)

# 4. Create embeddings and store them in FAISS vectorstore
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_EMBEDDING_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_EMBEDDING_API_KEY"),
    openai_api_version=os.getenv("AZURE_OPENAI_EMBEDDING_VERSION"),
    azure_deployment="text-embedding-ada-002"
)



from langchain_community.vectorstores import FAISS

BATCH_SIZE = 64
medical_vectorstore = None

for start in range(0, len(medical_docs), BATCH_SIZE):
    batch = medical_docs[start:start + BATCH_SIZE]

    if medical_vectorstore is None:
        medical_vectorstore = FAISS.from_documents(
            documents=batch,
            embedding=embeddings
        )
    else:
        medical_vectorstore.add_documents(batch)

    print(f"Processed {min(start + BATCH_SIZE, len(medical_docs))}/{len(medical_docs)} documents")

medical_vectorstore.save_local("faiss_index")

print("Batch ingestion completed successfully!")
print("Total documents indexed:", medical_vectorstore.index.ntotal)

