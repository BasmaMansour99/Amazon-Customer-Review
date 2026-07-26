# -*- coding: utf-8 -*-

# ==========================================
# Import Libraries
# ==========================================
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# ==========================================
# Build Grounded Prompt
# ==========================================
def build_grounded_prompt(query, context_text):
    """
    Strict, context-only prompt.
    Answers only from retrieved Amazon customer reviews.
    """

    if not context_text.strip():
        return None

    return f"""You are AutoAnalyst AI, an assistant that answers questions about
Amazon products using ONLY the customer review excerpts provided below.

Rules:
- Answer strictly using the information contained in the reviews below.
- Do not use outside knowledge or make assumptions beyond what is written.
- If the reviews do not contain enough information to answer the question,
  respond with exactly this sentence and nothing else:
  Insufficient information.
- Keep the answer brief and directly grounded in the review text.
- Where useful, mention which product(s) the answer is based on.

Question:
{query}

Customer review excerpts:
{context_text}

Answer:"""

# ==========================================
# Test Prompt Creation
# ==========================================
print("=== Test 1: Valid Prompt Creation ===")
query = "Do customers recommend this product?"

context_text = """
Customers like the battery life and screen quality.
Some users reported issues with charging.
"""

prompt = build_grounded_prompt(query, context_text)
print(prompt)

# ==========================================
# Test Empty Context
# ==========================================
print("\n=== Test 2: Empty Context Test ===")
empty_prompt = build_grounded_prompt("What is the product quality?", "")
print(f"Result for empty context: {empty_prompt}")