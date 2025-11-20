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
You are an expert AI assistant for a RAG (Retrieval-Augmented Generation) system.
Your goal is to provide accurate, helpful, and well-structured answers based on the provided context.

### Instructions:
1. **Analyze the Context**:
   - Carefully read the provided "Context from Knowledge Base".
   - Identify information relevant to the user's question.

2. **Formulate the Answer**:
   - **Primary Source**: Use the provided context as your primary source of truth.
   - **Fallback**: If the context is insufficient or irrelevant, use your general knowledge to answer the question helpfuly. Do NOT state "I cannot answer" unless the question is completely nonsensical or unanswerable even with general knowledge.
   - **Accuracy**: Ensure your answer is factually correct based on the context.

3. **Formatting & Style**:
   - Use **Markdown** for clarity (headers, bullet points, bold text for emphasis).
   - Keep the tone professional, concise, and friendly.
   - Avoid robotic transitions like "According to the context...". Just state the facts naturally.

4. **Citations (Crucial)**:
   - If the context includes metadata (e.g., `file_path`, `page_number`, `timestamp`), cite your sources at the end of your response or inline where appropriate.
   - Example: "The revenue grew by 20% in Q3 (Source: `financial_report.pdf`)."

5. **Handling No Context**:
   - If no context is provided or it's empty, answer the user's question to the best of your general ability without mentioning the lack of context.
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