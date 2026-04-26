import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to .env or set it in the environment."
    )

pdf_path = Path(__file__).parent / "sample_data.pdf"
# load the document
loader = PyPDFLoader(pdf_path)
# read the document and store it in a list of documents
docs = loader.load()
# print(docs[3])

# chunking the document into smaller pieces
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(docs)

# vector embedding
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
)

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning-rag",
)

print("Document loaded and indexed successfully!")