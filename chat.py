import os
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

openai_client = OpenAI()  # Initialize the OpenAI client

openai_api_key = os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY")
if not openai_api_key:
    raise RuntimeError(
        "OPENAI_API_KEY is required. Add it to .env or set it in the environment."
    )


embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=openai_api_key,
)

vector_db= QdrantVectorStore.from_existing_collection(
    embedding=embedding_model,
    url="http://localhost:6333",
    collection_name="learning-rag",
)

#Take the user input

user_query = input("Enter your question: ")

#returns relevent chunks from the vector database based on the user query
search_results = vector_db.similarity_search(user_query, k=3)


context = "\n\n\n".join([f"Page Content: {result.page_content}\n" for result in search_results])

SYSTEM_PROMPT = """You are a helpful AI assistant for answering questions based on the provided context. Use the following retrieved chunks to answer the question. context retrived from the a PDF file along with the page_content and page number. If the answer is not found in the provided context, say you don't know. Always use the provided context to answer the question. Do not make up answers.
 If you don't know the answer, say you don't know.
 Context:
 {context}"""

response = openai_response = openai_client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": user_query}
    ]
)

print(f"🤖:{response.choices[0].message.content}")