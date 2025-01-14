from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts.prompt import PromptTemplate
from langchain_pinecone import PineconeVectorStore
from supabase.lib.client_options import ClientOptions
from supabase.client import create_async_client, Client
import os
from dotenv import load_dotenv
import asyncio

load_dotenv()

# Supabase setup
supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")



# Langchain and Pinecone setup
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
document_vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX, embedding=embeddings)
retriever = document_vectorstore.as_retriever()
llm = ChatOpenAI(temperature=0.7, model_name="gpt-4")

async def process_message(supabase: Client, payload):
    """Process new messages and generate responses for those starting with /ask"""
    try:
        print(f"Received message: {payload}")
        message = payload['new']
        content = message.get('content', '')
        
        # Only process messages that start with /ask
        if not content.startswith('/ask'):
            return
            
        query = content[4:].strip()
        print(f"Processing query: {query}")
        
        context = retriever.invoke(query)
        print(f"Retrieved context: {context}")
        
        template = PromptTemplate(
            template="Answer this question: {query}\nUsing this context: {context}\nProvide a clear and concise response.",
            input_variables=["query", "context"]
        )
        prompt_with_context = template.invoke({"query": query, "context": context})
        
        response = llm.invoke(prompt_with_context)
        print(f"Generated response: {response.content}")
        
        await send_response(supabase, message, response.content)
        
    except Exception as e:
        print(f"Error processing message: {e}")
        import traceback
        print(traceback.format_exc())

async def send_response(supabase: Client, original_message, response_content):
    """Send a response back to the channel"""
    try:
        data = {
            "content": response_content,
            "user_id": "bot",
            "created_at": "now()",
        }
        
        if original_message.get('channel_id'):
            data["channel_id"] = original_message['channel_id']
        
        print(f"Sending response with data: {data}")
        
        result = await supabase.table('messages').insert(data).execute()
        print(f"Response sent: {result}")
        
    except Exception as e:
        print(f"Error sending response: {e}")
        import traceback
        print(traceback.format_exc())

async def main():
    """Main function to handle real-time message updates"""
    try:
        print("Starting chatbot service...")
        
        # Initialize async Supabase client
        supabase = await create_async_client(
            supabase_url,
            supabase_key,
            options=ClientOptions(
                postgrest_client_timeout=30,
                storage_client_timeout=30
            )
        )
        
        # Connect before creating or subscribing to channels
        await supabase.realtime.connect()

        channel = supabase.channel('messages')
        
        async def handle_insert(payload):
            print(f"Received payload: {payload}")
            await process_message(supabase, payload)
        
        
        await channel.subscribe()
        channel.on_postgres_changes(
            event='INSERT',
            schema='public',
            table='messages',
            callback=handle_insert
        )
        print("Successfully subscribed to messages channel")
        
        # Keep the connection alive
        while True:
            await asyncio.sleep(1)
            
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main()) 