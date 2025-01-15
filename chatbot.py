from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, UUID4
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from dotenv import load_dotenv
from supabase.client import create_client, Client
import os
from typing import Optional
import logging
import uuid

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Pydantic models for request/response
class Message(BaseModel):
    content: str
    channel_id: UUID4
    user_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "content": "What are the recent messages?",
                "channel_id": "123e4567-e89b-12d3-a456-426614174000",  # Example UUID
                "user_id": None
            }
        }

class ChatResponse(BaseModel):
    response: str
    channel_id: UUID4

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
        
    except Exception as e:
        logger.error(f"Failed to initialize services: {str(e)}")
        raise

def fetch_messages_from_db():
    if not supabase:
        raise ValueError("Supabase client not initialized")
        
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

async def store_bot_response(content: str, channel_id: UUID4):
    if not supabase:
        raise ValueError("Supabase client not initialized")
        
    try:
        # First verify the channel exists
        channel_check = supabase.from_("channels").select("id").eq("id", str(channel_id)).execute()
        if not channel_check.data:
            raise HTTPException(status_code=404, detail=f"Channel {channel_id} not found")

        # Get bot user
        bot_user = supabase.from_("users").select("id").eq("username", "bot").execute()
        
        logger.info(f"bot_user: {bot_user.data}")

        # Insert the message
        response = supabase.from_("messages").insert({
            "content": content,
            "channel_id": str(channel_id),
            "user_id": bot_user.data[0]["id"]
        }).execute()
        
        logger.info(f"Successfully stored bot response in channel {channel_id}")
        return response
        
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Failed to store bot response: {str(e)}")
        # Log the full error details for debugging
        logger.error(f"Full error details: {repr(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

def query_vectorstore(query: str, channel_id: UUID4):
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

@app.on_event("startup")
async def startup_event():
    await initialize_services()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(message: Message):
    if not vectorstore:
        raise HTTPException(status_code=503, detail="Service not fully initialized")
        
    try:
        # Get bot's response
        response = query_vectorstore(message.content, message.channel_id)
        
        # Store the bot's response in the database
        await store_bot_response(response, message.channel_id)
        
        return ChatResponse(response=response, channel_id=message.channel_id)
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

@app.get("/debug/channel/{channel_id}")
async def debug_channel(channel_id: UUID4):
    if not supabase:
        raise HTTPException(status_code=503, detail="Supabase not initialized")
    
    try:
        # Check channel
        channel = supabase.from_("channels").select("*").eq("id", str(channel_id)).execute()
        
        # Check if bot user exists
        bot_user = supabase.from_("users").select("*").eq("id", "bot").execute()
        
        return {
            "channel_exists": len(channel.data) > 0,
            "channel_data": channel.data,
            "bot_user_exists": len(bot_user.data) > 0,
            "bot_user_data": bot_user.data
        }
    except Exception as e:
        logger.error(f"Debug endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 