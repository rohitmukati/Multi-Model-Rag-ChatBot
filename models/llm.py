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
SYSTEM_PROMPT = """
You are a high-quality RAG (Retrieval-Augmented Generation) assistant.

Your behavior rules:

1. If context is provided:
   - First check the context for relevant information.
   - If the context contains the answer, use ONLY the context.
   - If the context does NOT contain the answer, then fall back to general knowledge and answer accurately.

2. If NO context is provided:
   - Answer the user's question normally with correct and concise information.

3. Never say:
   - "I cannot answer because it is outside context."
   - "I don't have enough information."
   unless the information truly does not exist anywhere, even in general world knowledge.

4. Always answer directly, clearly, and helpfully.
   No over-explanations. No unnecessary disclaimers.

5. If context has metadata (file paths, timestamps, etc.), optionally cite useful sources when meaningful.

Your goal: Always give the best possible answer — either from context or from general knowledge — whichever is relevant and accurate.
"""



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