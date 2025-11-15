"""
LLM Integration using FREE Gemini API
Put this file inside: models/llm.py
"""

import os
import sys

# Add project root to Python path
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Gemini import with safety check
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "google-generativeai is not installed.\n"
        "Install it using:\n\n"
        "   pip install google-generativeai\n"
    )

# Get API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("⚠ WARNING: GEMINI_API_KEY missing in .env file")
    print("Get free API key: https://ai.google.dev/")
    GEMINI_API_KEY = "DUMMY_KEY"   # still allows import + prevents crash

# Configure API
genai.configure(api_key=GEMINI_API_KEY)

# Select model
DEFAULT_MODEL = "gemini-2.0-flash-exp"

try:
    model = genai.GenerativeModel(DEFAULT_MODEL)
except Exception as e:
    raise RuntimeError(f"Failed to initialize Gemini model: {e}")


# ==========================
# MAIN GENERATION FUNCTION
# ==========================

def generate_response(query, context):
    """
    Generates an LLM response using RAG retrieved context.
    
    Args:
        query: The user’s question.
        context: Combined retrieved content passed as a string.

    Returns:
        Formatted LLM answer string.
    """

    prompt = f"""
You are a helpful AI assistant with access to a knowledge base.

### CONTEXT:
{context}

### USER QUESTION:
{query}

### INSTRUCTIONS:
- Answer **only** using the context.
- If context does not contain the answer, respond:
  "I don't have enough information based on the stored knowledge."
- Be concise, accurate, and avoid hallucinations.
- If multiple sources are retrieved, merge info cleanly.
"""

    try:
        response = model.generate_content(prompt)
        return response.text.strip()

    except Exception as e:
        return f"LLM Error: {str(e)}"


# ==========================
# MODEL TEST FUNCTION
# ==========================

def test_llm():
    """Checks if Gemini is working correctly."""

    print("🔍 Listing available Gemini models:")
    try:
        for m in genai.list_models():
            if "generateContent" in m.supported_generation_methods:
                print("  -", m.name)
    except Exception as e:
        print("  Failed to list models:", e)

    print("\n🧪 Running test query...")

    dummy_context = "Python language was created by Guido van Rossum."
    dummy_query = "Who created Python?"

    response = generate_response(dummy_query, dummy_context)

    print("\n✅ TEST RESPONSE:\n", response)
    return response


# Allow direct testing
if __name__ == "__main__":
    test_llm()
