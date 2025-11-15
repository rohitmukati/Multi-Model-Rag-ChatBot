# Auto-generated file
"""
LLM Integration using FREE Gemini API
Replace your models/llm.py with this code
"""

import google.generativeai as genai
from config.settings import config
import os
from dotenv import load_dotenv
load_dotenv()


# Configure Gemini
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

# Use Gemini 1.5 Flash (FREE, Fast)
model = genai.GenerativeModel('gemini-1.5-flash')

def generate_response(query, context):
    """
    Generate response using Gemini with retrieved context
    
    Args:
        query: User's question
        context: Retrieved text chunks from vector DB
    
    Returns:
        Generated answer
    """
    
    prompt = f"""You are a helpful AI assistant with access to a knowledge base.
    
Context from knowledge base:
{context}

User Question: {query}

Instructions:
- Answer based on the context provided above
- If context doesn't contain the answer, say "I don't have enough information"
- Be concise and accurate
- Cite sources when possible

Answer:"""
    
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"


# Test function
def test_llm():
    """Test if Gemini is working"""
    test_context = "Python is a high-level programming language. It was created by Guido van Rossum."
    test_query = "Who created Python?"
    
    print("🧪 Testing Gemini LLM...")
    response = generate_response(test_query, test_context)
    print(f"✅ Response: {response}")
    return response


if __name__ == "__main__":
    # Run this file directly to test
    test_llm()