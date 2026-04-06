import asyncio
import json
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"

async def test_extraction(user_msg, options):
    options_str = "\n".join([f"- {opt}" for opt in options])
    prompt = f"""
    You are an expert intent extraction engine for a supermarket.
    The user is trying to select an item from a list of previously suggested options.
    
    USER MESSAGE: "{user_msg}"
    
    SUGGESTED OPTIONS:
    {options_str}
    
    YOUR TASKS:
    1. Identify which option the user is selecting.
    2. Handle minor typos (e.g., "oxtial" -> "Oxtail").
    3. Handle ordinal references (e.g., "the first one", "the second", "1st").
    4. Handle descriptive references (e.g., "the fresh one", "the boneless one").
    5. Match the user's intent to the EXACT name from the SUGGESTED OPTIONS list.
    6. If the user message is generic (e.g., "yes please", "sure"), and there is only one logical option, pick it.
    7. If multiple products match or it's genuinely ambiguous, return null.
    8. If it's a "no" or rejection, return null.
    
    OUTPUT FORMAT (JSON only):
    {{
        "selected_inventory": "exact_name_from_options_or_null"
    }}
    """
    
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": user_msg}],
        "response_format": {"type": "json_object"},
        "temperature": 0.1
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
        return response.json()["choices"][0]["message"]["content"]

async def main():
    test_cases = [
        ("the oxtial", ["Chicken", "Pork", "Oxtail", "Beef"]),
        ("the first one", ["Jasmine Rice", "Basmati Rice", "Brown Rice"]),
        ("ooooh the oxtails how much are they", ["Chicken", "Pork", "Oxtail", "Beef"]),
        ("the fresh one", ["Oxtail - Canned", "Oxtail - Fresh", "Oxtail - Frozen"]),
    ]
    
    for msg, opts in test_cases:
        print(f"\nTesting: '{msg}' against {opts}")
        res = await test_extraction(msg, opts)
        print(f"Result: {res}")

if __name__ == "__main__":
    asyncio.run(main())
