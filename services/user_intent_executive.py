# services/user_intent_executive.py
from datasets.data import inventory_df, recipes_df, carts
from models.llm_classes import IntentQuery, Entity, IntentExecution, LLMResponse
import httpx
import json
import re
from typing import Union
from services.llm_logic import (prepare_stock_response, prepare_check_recipe_availability_response,
                                prepare_get_recipe_response, prepare_recommend_recipe_response, prepare_budget_recipe_response,
                                prepare_cart_response)
from intents.get_recipe import get_recipe_data
import time
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG — swap keys/model here
# ─────────────────────────────────────────────

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
DEEPSEEK_URL     = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b"

USE_DEEPSEEK     = True   # flip to False to force local Ollama


# ─────────────────────────────────────────────
#  DEEPSEEK CALL
# ─────────────────────────────────────────────

async def call_deepseek(system_prompt: str, user_message: str, chat_history: list = []) -> str:
    """Call DeepSeek API — fast, handles intent + response generation."""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_message})

    payload = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 500,
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        start = time.perf_counter()
        try:
            response = await client.post(DEEPSEEK_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            duration = time.perf_counter() - start
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            return None   # triggers Ollama fallback


# ─────────────────────────────────────────────
#  OLLAMA CALL (local fallback)
# ─────────────────────────────────────────────

async def call_ollama(system_prompt: str, user_message: str, chat_history: list = []) -> str:
    """Local Ollama fallback — slower but works offline."""
    custom_timeout = httpx.Timeout(160.0, connect=60.0)
    
    # Construct prompt from history for Ollama
    history_str = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"\n{role}: {msg['content']}"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": system_prompt + history_str + "\nUser: " + user_message,
        "stream": False,
        "format": "json",
        "keep_alive": -1,
        "options": {
            "num_predict": 150,
            "temperature": 0.1,
            "num_ctx": 512,
            "top_k": 10,
            "top_p": 0.5,
            "repeat_penalty": 1.0,
        }
    }

    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        start = time.perf_counter()
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            #duration = time.perf_counter() - start
            #tokens = data.get("eval_count", 0)
            return data.get("response")
        except httpx.TimeoutException:
            return "Error: The AI took too long to think."
        except Exception as e:
            return f"Error: {str(e)}"


# ─────────────────────────────────────────────
#  UNIFIED LLM CALL
# ─────────────────────────────────────────────

async def message_to_llm(system_prompt: str, user_message: str, chat_history: list = []) -> str:
    """Try DeepSeek first, fall back to Ollama if it fails or is disabled."""
    if USE_DEEPSEEK:
        result = await call_deepseek(system_prompt, user_message, chat_history)
        if result is not None:
            return result
    return await call_ollama(system_prompt, user_message, chat_history)


# ─────────────────────────────────────────────
#  INTENT PROMPT
# ─────────────────────────────────────────────

def fetch_intent():
    return """
        You are a specialized JSON extraction engine for a grocery store.
        Your goal is to parse user input into a structured format matching the provided schema.

        INTENTS:
        - "stock_check": User asks if an item is available or its price.
            Example: "Do you have milk?" or "How much is the bread?" or "Got any apples?" or "I need chicken" or "I want eggs"
        - "get_recipe": User asks for instructions or ingredients for a specific dish.
            Example: "How do I make pancakes?" or "What do I need for spaghetti?"
        - "add_to_cart": User clearly wants to put an item in their shopping basket.
            Example: "Add 2 cartons of eggs to my cart." or "Yes" if responding to "Do you want to add chicken to your cart?"
        - "remove_from_cart": User wants to remove an item from their shopping basket.
            Example: "Remove the milk from my cart"
        - "view_cart": User wants to see what's in their cart or check their total.
            Example: "What's in my cart?" or "Show my basket"
        - "check_recipe_availability": User wants to know if the store has ingredients for a recipe.
            Example: "Can I make pancakes with what you have?"
        - "recommend_recipe": User is looking for meal ideas based on available ingredients.
            Example: "What can I make with chicken and rice?"
        - "recipe_from_cart_items": User wants recipes based on items already in their cart.
            Example: "What can I make with what's in my cart?" 
                      "Give me recipes for my cart items"
                      "What recipes use what I already have"
        - "budget_recipe_suggestion": User wants recipe suggestions based on a specific budget.
            Example: "What can I make for under $1000?"
        - "general_chat": User is greeting you, being vague, or making a comment.
            Example: "Hi there!" or "What's up?" or "That sounds good."
        - "take_me_to_cart": User wants to view their cart page.
            Example: "Take me to my cart" or "I want to see my basket"
        - "terms_and_conditions": User wants to see the store's policies.
            Example: "What are your terms and conditions?" or "Tell me about your privacy policy"
                    - "What is your return policy?" or "How do I return an item?"
                    - "How long do I have to return something?" or "Can I return fresh produce?"
                    - "What is your refund process?" or "Do you give refunds for opened items?"

        ENTITIES:
        - product_name: The item (e.g., "apples").
        - recipe_name: The dish (e.g., "mac and cheese").
        - category_hint: Based on the item, predict the store category.
        - quantity: number if the user states it
        - budget: monetary amount if mentioned

        RULES:
        1. Return ONLY valid JSON. No conversational text.
        2. If an entity is missing, use null.
        3. If the intent is unclear, use "general_chat".
        4. If the user says something like "yes", "sure", "ok", or "no", "not really" in response to an assistant question, map it to the logical intent.
           - If responding to "Would you like to add this to your cart?", "yes" -> "add_to_cart".
           - If responding to "Would you like a recipe for this?", "yes" -> "get_recipe".
        5. If the user mentions a monetary value or budget, intent must be "budget_recipe_suggestion".
        6. If the user asks for a fresh ingredient (fruit/veg), category_hint MUST be "Produce".
        8. If the user says "I need X", "I want X", "I am looking for X", or asks "got X?" / "you have X?", intent is "stock_check".
        8. If user mentions products and intent is "recommend_recipe", list them in product_name.
        9. Look at the conversation history to resolve pronouns like "that", "it", "the first one".

        OUTPUT FORMAT:
        {
            "intent": "intent_name",
            "entities": {
                "product_name": "name" or null,
                "quantity": number or null,
                "recipe_name": "name" or null,
                "budget": number or null,
                "category_hint": "category" or null
            }
        }
        """


# ─────────────────────────────────────────────
#  RESPONSE PARSING
# ─────────────────────────────────────────────

def llm_response_extraction(llm_data: Union[str, dict]):
    try:
        parsed_json = json.loads(llm_data) if isinstance(llm_data, str) else llm_data
        #print("Parsed JSON:", parsed_json)

        return IntentExecution(
            intent=parsed_json.get("intent", "general_chat"),
            entities=Entity(
                product_name=parsed_json.get("entities", {}).get("product_name"),
                quantity=parsed_json.get("entities", {}).get("quantity"),
                recipe_name=parsed_json.get("entities", {}).get("recipe_name"),
                servings=parsed_json.get("entities", {}).get("servings"),
                product_codes=parsed_json.get("entities", {}).get("product_codes"),
                budget=parsed_json.get("entities", {}).get("budget"),
                category_hint=parsed_json.get("entities", {}).get("category_hint"),
            )
        )
    except Exception as e:
        #print(f"Parsing Error: {e} | Raw: {llm_data}")
        return IntentExecution(intent="general_chat", entities=Entity())


# ─────────────────────────────────────────────
#  MAIN INTENT ROUTER
# ─────────────────────────────────────────────

async def intent_detection(llm_data: str, user_message: str, user_id: str, chat_history: list = []):
    intent_possibilities = ["stock_check", "get_recipe", "add_to_cart", "remove_from_cart", "view_cart",
                            "check_recipe_availability", "recommend_recipe",
                            "budget_recipe_suggestion", "general_chat", "take_me_to_cart", "terms_and_conditions", "recipe_from_cart_items"]

    
    execution = llm_response_extraction(llm_data)
    intent    = execution.intent
    entities  = execution.entities
    #print(f"Extracted Intent: {intent} | Entities: {entities}")

    if intent not in intent_possibilities:
        intent = "general_chat"

    # Fallback: treat direct shopping requests as stock checks when intent falls back to general chat.
    if intent == "general_chat":
        normalized = user_message.lower()
        stock_keywords = re.compile(r"\b(i need|need|i want|want|looking for|looking to buy|find|got any|have any|have|do you have)\b")
        recipe_keywords = re.compile(r"\b(recipe|cook|make|how to|how do i|what can i make|suggest|budget|under \$|under \d+)\b")
        if stock_keywords.search(normalized) and not recipe_keywords.search(normalized):
            intent = "stock_check"
            #print(f"Overriding intent to stock_check based on fallback heuristic for: {user_message}")
        else:
            words = re.findall(r"\w+", normalized)
            ignore_short = {"yes", "y", "yep", "yeah", "no", "nope", "nah", "thanks", "thank", "thank you", "ok", "okay", "sure", "please", "hi", "hello", "bye", "goodbye"}
            if 0 < len(words) <= 3 and not recipe_keywords.search(normalized) and any(w not in ignore_short for w in words):
                intent = "stock_check"
                #print(f"Overriding intent to stock_check for short shopping request: {user_message}")
    llm_response = ""

    match intent:
        case "stock_check":
            #print("Stock check logic triggered.")
            # Handled fully in Python — no LLM call needed
            system_prompt = await prepare_stock_response(entities, user_message)

            # Send the stock response to the llm to message to the user 
            llm_response = await message_to_llm(system_prompt, " ", chat_history)

            # Return the llm response
            return llm_response

        case "get_recipe":
            # Check if the intent is "get_recipe"
            #Check recipe availability
            system_prompt = await prepare_get_recipe_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)
            #print("LLM response:", llm_response)

            return llm_response

        case "add_to_cart":
            # Call the cart response logic
            system_prompt = await prepare_cart_response(entities, user_message, user_id)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "remove_from_cart":
            from services.llm_logic import prepare_remove_from_cart_response
            system_prompt = await prepare_remove_from_cart_response(entities, user_message, user_id)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "view_cart":
            from services.llm_logic import prepare_view_cart_response
            system_prompt = await prepare_view_cart_response(user_id)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)
            
            return llm_response
        case "check_recipe_availability":
            # Handle recipe availability
            system_prompt = await prepare_check_recipe_availability_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)

            #print("Recipe LLM response:", llm_response)
            return llm_response
        
        case "recommend_recipe":
            system_prompt = await prepare_recommend_recipe_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "recipe_from_cart_items":
            # Fetch cart items and pass as context
            user_cart = carts.get(user_id, [])
            cart_items = [item['product_name'] for item in user_cart]
            
            if not cart_items:
                return json.dumps({"message": "Your cart is empty! Add some items first."})
            
            # Create a fake entity with cart items
            entities.product_name = " and ".join(cart_items)
            
            from intents.recommend_recipe import recommend_recipe as get_recommendations
            recommendations = get_recommendations(entities, user_id, user_message)
            
            system_prompt = await prepare_recommend_recipe_response(entities, user_message, recommendations)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)
            return llm_response
        
        case "budget_recipe_suggestion":
            system_prompt = await prepare_budget_recipe_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "take_me_to_cart":
            return json.dumps({"message": "Sure! Redirecting you to your cart now.",
                               'action': 'redirect_to_cart',
                               'redirect_url': '/cart'})

        case "terms_and_conditions":
            # Use RAG to fetch the terms and conditions from a local file or database
            from intents.terms_and_conditions import terms_and_conditions, prepare_terms_and_conditions_response
            system_prompt = await prepare_terms_and_conditions_response(user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response

        
        case "general_chat":
            from intents.general_chat import prepare_general_chat_response
            system_prompt = await prepare_general_chat_response(user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response

    return llm_response