from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv
from supabase.client import create_client
import os
from typing import Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic models for request/response
class Message(BaseModel):
    content: str

class ChatResponse(BaseModel):
    content: str

# Global variables for initialized services
vectorstore = None
embeddings = None
supabase = None
PINECONE_INDEX = None

def load_environment():
    logger.info("Loading environment variables...")
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
            logger.error(f"Missing environment variable: {var}")
            raise ValueError(f"Missing environment variable: {var}")
        os.environ[var] = val
    logger.info("Environment variables loaded successfully")
    return os.getenv("PINECONE_INDEX")

def initialize_supabase():
    logger.info("Initializing Supabase client...")
    supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL") or os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY") or os.getenv("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        logger.error("Missing Supabase credentials")
        raise ValueError("Missing Supabase credentials")
    return create_client(supabase_url, supabase_key)

async def initialize_services():
    global vectorstore, embeddings, supabase, PINECONE_INDEX
    try:
        logger.info("Starting service initialization...")
        
        # Load environment variables
        PINECONE_INDEX = load_environment()
        
        # Initialize Supabase
        supabase = initialize_supabase()
        
        # Initialize embeddings
        logger.info("Initializing OpenAI embeddings...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        
        # Fetch messages and initialize vectorstore
        logger.info("Fetching messages from database...")
        documents = fetch_messages_from_db()
        logger.info(f"Found {len(documents)} messages")
        
        logger.info("Initializing Pinecone vectorstore...")
        vectorstore = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            index_name=PINECONE_INDEX
        )
        logger.info("Service initialization completed successfully")
        
        # Update all messages to be vectorized
        logger.info("Updating all messages to be vectorized...")
        update_messages_to_vectorized()
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

def update_messages_to_vectorized():
    if not supabase:
        raise ValueError("Supabase client not initialized")
        
    supabase.from_("messages").update({"vectorized": True}).neq("content", None).execute()

def fetch_messages_from_db():
    if not supabase:
        raise ValueError("Supabase client not initialized")
        
    response = supabase.from_("messages").select(
        "content, created_at, channel_id, vectorized, users(username)"
    ).neq("content", None).eq("vectorized", False).order("created_at", desc=True).execute()
    
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

def handle_fake_response(query: str) -> str:
    """Handle queries starting with /fake"""
    # Extract username and query from /fake command
    actual_query = query.removeprefix("/fake").strip()
    username = actual_query.split()[0]
    actual_query = actual_query[len(username):].strip()

    # Get relevant context from vectorstore
    if not vectorstore:
        raise ValueError("Vectorstore not initialized")
    
    retriever = vectorstore.as_retriever()
    context = retriever.invoke(actual_query)

    template = PromptTemplate(
        template="""Based on the following chat history, generate a response as if you are {username} continuing the conversation.
        Format your response as '{username}: [response]'
        
        Chat history context:
        {context}
        
        Query to respond to: {query}
        
        Remember to stay in character as {username} and reference relevant details from the chat history.
        """,
        input_variables=["query", "username", "context"]
    )
    
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    prompt = template.invoke({
        "query": actual_query,
        "username": username,
        "context": context
    })
    results = llm.invoke(prompt)
    
    return results.content

def handle_file_query(query: str) -> str:
    """Handle queries about files in Supabase storage"""
    if not supabase:
        raise ValueError("Supabase client not initialized")
    
    # TODO: Implement file vectorization and querying from Supabase storage
    # This would involve:
    # 1. Fetching files from Supabase storage 'uploads' folder
    # 2. Processing/vectorizing their content
    # 3. Querying against those vectors
    
    return "File querying functionality is not implemented yet"

def handle_normal_query(query: str) -> str:
    """Handle regular queries using existing vectorstore"""
    if not vectorstore:
        raise ValueError("Vectorstore not initialized")
        
    retriever = vectorstore.as_retriever()
    context = retriever.invoke(query)
    
    template = PromptTemplate(
        template="Based on the chat history provided, please answer the following question: {query}\n\nRelevant chat context: {context}",
        input_variables=["query", "context"]
    )
    prompt_with_context = template.invoke({"query": query, "context": context})
    
    llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")
    results = llm.invoke(prompt_with_context)
    
    return results.content

async def update_vectorstore():
    """Update vectorstore with any new messages before processing query"""
    try:
        logger.info("Fetching new messages for vectorstore update...")
        documents = fetch_messages_from_db()
        if documents:
            logger.info(f"Found {len(documents)} new messages to vectorize")
            # Add new documents to existing vectorstore
            await vectorstore.aadd_documents(documents)
            # Mark messages as vectorized
            update_messages_to_vectorized()
            logger.info("Vectorstore update completed")
    except Exception as e:
        logger.error(f"Error updating vectorstore: {str(e)}")
        raise

async def process_query(query: str) -> str:
    """Process the query based on its prefix"""
    # Update vectorstore before processing query
    await update_vectorstore()
    
    query = query.strip()
    
    if query.startswith("/fake"):
        return handle_fake_response(query)
    elif query.lower().startswith("/file"):
        return handle_file_query(query)
    else:
        return handle_normal_query(query)

@app.on_event("startup")
async def startup_event():
    await initialize_services()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: Message):
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
        
    try:
        response = await process_query(message.content)
        
        return ChatResponse(content=response)
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    if not vectorstore:
        return {"status": "initializing"}
    return {"status": "healthy"}