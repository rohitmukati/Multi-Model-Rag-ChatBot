"""
LLM Integration using FREE Gemini API
Final Production-Ready Code for models/llm.py
"""

import os
import sys
from dotenv import load_dotenv
from google import genai

# Load environment variables
load_dotenv()

# Initialize Gemini Client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL = "gemini-2.5-flash"

# System prompt for RAG assistant
SYSTEM_PROMPT = """You are an intelligent RAG (Retrieval-Augmented Generation) assistant.

Your responsibilities:
1. Answer questions based ONLY on the provided context from the knowledge base
2. If the context doesn't contain relevant information, clearly state: "I don't have enough information in the knowledge base to answer this question."
3. Always cite sources when available (mention file names, page numbers, etc.)
4. Be concise, accurate, and helpful
5. If asked about topics outside the context, politely redirect to knowledge base content
6. If user asks for general knowledge, respond with short and simple answer.

Remember: Never make up information. Only use the context provided."""


def generate_response(query: str, context: str = "") -> str:
    
    if context:
        # RAG mode: Answer based on retrieved context
        prompt = f"""{SYSTEM_PROMPT}

Context from Knowledge Base:
---
{context}
---
User Question: {query}
Answer (based on the context above):"""

    else:
        prompt = f"""{SYSTEM_PROMPT}
User Question: {query}
Answer:"""
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text.strip()
    
    except Exception as e:
        return f"Error generating response: {str(e)}"


def generate_simple_response(prompt: str) -> str:
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    test_query = "What is the capital of France?"
    print("Test Query Response:")
    print(generate_response(test_query))