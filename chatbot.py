from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv
from supabase.client import create_client, Client
import os

def load_environment():
    load_dotenv()
    required_vars = [
        "PINECONE_API_KEY",
        "LANGCHAIN_API_KEY",
        "LANGCHAIN_TRACING_V2",
        "LANGCHAIN_PROJECT",
        "PINECONE_INDEX",
        "OPENAI_API_KEY"
    ]
    for var in required_vars:
        val = os.getenv(var)
        if val is None:
            raise ValueError(f"Please set the environment variable {var} in your .env file.")
        os.environ[var] = val
    return os.getenv("PINECONE_INDEX")

def fetch_messages_from_db():
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    supabase: Client = create_client(supabase_url, supabase_key)

    # Single query to get messages with usernames
    response = supabase.from_("messages").select(
        "content, created_at, channel_id, users(username)"
    ).neq("content", None).order("created_at", desc=True).execute()
    
    documents = []
    for msg in response.data:
        content = msg.get("content")
        created_at = msg.get("created_at")
        channel_id = msg.get("channel_id")
        username = msg.get("users", {}).get("username", "unknown")
        
        if not all([content, created_at, channel_id]):
            continue
            
        formatted_content = (
            f"Channel: #{channel_id}\n"
            f"Author: @{username}\n"
            f"Time: {created_at}\n"
            f"Message: {content}"
        )
        
        doc = Document(
            page_content=formatted_content,
            metadata={
                "username": username,
                "channel": str(channel_id),
                "timestamp": created_at
            }
        )
        documents.append(doc)
    
    return documents

def initialize_vectorstore(documents):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = PineconeVectorStore.from_documents(
        documents=documents,
        embedding=embeddings,
        index_name=PINECONE_INDEX
    )
    return vectorstore, embeddings

def query_vectorstore(query, vectorstore, embeddings):
    retriever = vectorstore.as_retriever()
    context = retriever.invoke(query)
    
    print("\nRelevant messages found:")
    print("------------------------")
    for doc in context:
        print(f"{doc.page_content}\n")
    print("------------------------")
    
    template = PromptTemplate(
        template="Based on the chat history provided, please answer the following question: {query}\n\nRelevant chat context: {context}",
        input_variables=["query", "context"]
    )
    prompt_with_context = template.invoke({"query": query, "context": context})
    
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    results = llm.invoke(prompt_with_context)
    
    return results.content

def main():
    global PINECONE_INDEX
    PINECONE_INDEX = load_environment()
    
    print("Fetching messages from database...")
    documents = fetch_messages_from_db()
    print(f"Found {len(documents)} messages")
    
    print("Initializing vector store...")
    vectorstore, embeddings = initialize_vectorstore(documents)
    print("Vector store initialized")
    
    print("\nChat History Query System")
    print("Enter your questions (or 'quit' to exit)")
    
    while True:
        query = input("\nYour question: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
            
        if not query:
            continue
            
        try:
            response = query_vectorstore(query, vectorstore, embeddings)
            print("\nResponse:", response)
        except Exception as e:
            print(f"Error processing query: {e}")

if __name__ == "__main__":
    load_dotenv()
    main() 