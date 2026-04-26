# RAG Document Q&A System

A Retrieval-Augmented Generation (RAG) system for question-answering on PDF documents using LangChain, OpenAI embeddings, and Qdrant vector database.

## Features

- Load and process PDF documents
- Split documents into manageable chunks with overlap
- Generate embeddings using OpenAI's text-embedding-3-small model
- Store and retrieve vectors in Qdrant database
- Ready for Q&A pipeline integration

## Prerequisites

- Python 3.8+
- OpenAI API key
- Qdrant server running locally (default: http://localhost:6333)

## Installation

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd rag-doc-qa
   ```

2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/Scripts/activate  # On Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env` (if available) or create `.env`
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your-api-key-here
     ```

## Usage

1. Place your PDF file (e.g., `sample_data.pdf`) in the project root.

2. Run the indexing script:

   ```bash
   python index.py
   ```

   This will:
   - Load the PDF
   - Split it into chunks
   - Generate embeddings
   - Store vectors in Qdrant

3. The system is now ready for Q&A queries (extend with retrieval and generation components).

## Project Structure

```
rag-doc-qa/
├── index.py              # Main indexing script
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (ignored by git)
├── .gitignore           # Git ignore rules
├── sample_data.pdf      # Sample PDF document
└── README.md            # This file
```

## Dependencies

- langchain: Framework for LLM applications
- langchain-openai: OpenAI integrations
- langchain-qdrant: Qdrant vector store
- langchain-community: Community loaders and tools
- qdrant-client: Qdrant database client
- python-dotenv: Environment variable management
- pypdf: PDF processing

## Configuration

- **Chunk size**: 1000 characters with 200 character overlap
- **Embedding model**: text-embedding-3-small
- **Qdrant URL**: http://localhost:6333
- **Collection name**: learning-rag
