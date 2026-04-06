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
from services.context_handler import get_user_state
from intents.recommend_recipe import is_recipe_selection


load_dotenv()

# ─────────────────────────────────────────────
#  CONFIG — swap keys/model here
# ─────────────────────────────────────────────

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "your-deepseek-api-key-here")
DEEPSEEK_URL     = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL   = "deepseek-chat"

OLLAMA_URL       = "http://localhost:11434/api/generate"
OLLAMA_MODEL     = "qwen2.5:1.5b"

GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "your-gemini-api-key-here")
GEMINI_URL       = "https://generativelanguage.googleapis.com/v1beta2/models/gemini-2.5-pro:generateContent"
GEMINI_MODEL     = "gemini-2.5-pro"

USE_DEEPSEEK     = True 


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
        "max_tokens": 1500,
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

    # --- FIX 1: DEFENSIVE PROGRAMMING ---
    # Ensure these are strings even if the caller passes None by mistake
    safe_system_prompt = system_prompt if system_prompt is not None else ""
    safe_user_message = user_message if user_message is not None else ""

    # Construct prompt from history for Ollama
    history_str = ""
    if chat_history: # Ensure chat_history isn't None either
        for msg in chat_history:
            # Safely get content, default to empty string if missing
            content = msg.get('content', '')
            role = "User" if msg.get("role") == "user" else "Assistant"
            history_str += f"\n{role}: {content}"

    # --- FIX 2: SAFER STRING BUILDING ---
    full_prompt = f"{safe_system_prompt}{history_str}\nUser: {safe_user_message}"

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": full_prompt,
        "stream": False,
        "format": "json",
        "keep_alive": -1,
        "options": {
            "num_predict": 1500,
            "temperature": 0.1,
            "num_ctx": 4096, # Increased context slightly for larger histories
            "top_k": 10,
            "top_p": 0.5,
            "repeat_penalty": 1.0,
        }
    }

    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        try:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            
            # --- FIX 3: STRIP LLM WHITESPACE ---
            llm_text = data.get("response", "")
            return llm_text.strip() if llm_text else ""
            
        except httpx.TimeoutException:
            return json.dumps({"message": "Error: The AI took too long to think.", "action_ready": False})
        except Exception as e:
            print(f"[OLLAMA ERROR] {str(e)}") # Log it so you can see it in your terminal
            return json.dumps({"message": f"Error: {str(e)}", "action_ready": False})

# ─────────────────────────────────────────────
#  UNIFIED LLM CALL
# ─────────────────────────────────────────────

async def message_to_llm(system_prompt: str, user_message: str, chat_history: list = []) -> str:
    """Try DeepSeek first, fall back to Ollama if it fails or is disabled."""
    if USE_DEEPSEEK:
        result = await call_deepseek(system_prompt, user_message, chat_history)
        print(f"DeepSeek result: {result}")  # Debug log
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
        - "shopping_list": User uploads a shopping list.
            Example: "I have a shopping list" or "Here is my shopping list"
        - "add_shopping_list_to_cart": User wants to add their shopping list to their cart.
            Example: "Add my shopping list to my cart" or "Add my list to my cart"
        - "inventory_check": User wants to know whats in the inventory
            Example: "What type of meat do you have?" or "What type of vegetables do you have?"
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
        - "checkout": User wants to checkout.
            Example: "Checkout" or "I want to checkout" or "I'm ready to checkout"
        - "terms_and_conditions": User wants to see the store's policies.
            Example: "What are your terms and conditions?" or "Tell me about your privacy policy"
                    - "What is your return policy?" or "How do I return an item?"
                    - "How long do I have to return something?" or "Can I return fresh produce?"
                    - "What is your refund process?" or "Do you give refunds for opened items?"
        - "product_location": User wants to know the location(aisle number) of a product in the physical store
            Example: "Where is the milk?" or "Where can I find the bread?" or "Where is the chicken?"
        - "store_info": User wants to know general information about the store (store hours, store contact, store payment methods, store delivery options)
            Example: "What time do you close?" or 
                    "What time do you open?" or 
                    "Where is the store?" or                    
                    "What is the phone number of the store?" or 
                    "What is the email address of the store?" or 
                    "How do I contact the store?" or 
                    "What payment methods do you accept?" or 
                    "Do you offer delivery?" or 
                    "Do you offer pickup?" or 
                    "Do you offer curbside pickup?"
        - "follow_up": User is responding to a previous question or selecting from a list
            Example: "Yes that one" or "the first one", "chicken" or "beef" or "oxtails" or "that one"
            
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
        9. If the user's message is short (1-3 words) AND the last assistant message contained a list or asked a question, set intent to "follow_up".
        10. If the user says something like "yes", "sure", "ok", "no", or a single item name ("oxtail", "chicken"), and the previous assistant message was asking for selection, set intent to "follow_up".
        11. If user says something vague like "what can i make" or asks for recipe suggestions without mentioning specific ingredients, set intent to "recipe_from_cart_items".
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
                            "budget_recipe_suggestion", "general_chat", "take_me_to_cart", "terms_and_conditions", "recipe_from_cart_items",
                            "inventory_check", "shopping_list", "add_shopping_list_to_cart", "store_info", "product_location", "checkout",
                            "follow_up"]

    
    execution = llm_response_extraction(llm_data)
    intent    = execution.intent
    entities  = execution.entities
    print(f"Extracted Intent: {intent} | Entities: {entities}")

    if intent not in intent_possibilities:
        intent = "general_chat"

    # Contextual Override: If the assistant just explicitly asked if the user wants to add items to their cart, 
    # treat their next short response (even if it's just a noun like "the boneless chicken") as an add_to_cart intent.
    # Contextual Overrides based on conversation history
    if len(chat_history) > 0:
        last_bot_msg = chat_history[-1].get("content", "").lower()
        
        # 1. Add to cart override
        if "add any of these to your cart" in last_bot_msg or "add either of these to your cart" in last_bot_msg or "add to your cart" in last_bot_msg:
            words = re.findall(r"\w+", user_message.lower())
            if len(words) <= 5 and intent in ["general_chat", "stock_check"]:
                intent = "add_to_cart"
                print(f"Overriding intent to add_to_cart based on conversation history.")
                
        # 2. Inventory check override (Selection Focus)
        elif any(phrase in last_bot_msg for phrase in ["which would you like to explore", "which one interests you", "from our list of", "which type", "refer to it by name or number"]):
            words = re.findall(r"\w+", user_message.lower())
            # Broaden to 15 words to capture "the oxtails how much are they"
            if len(words) <= 15 and intent in ["general_chat", "stock_check", "add_to_cart", "follow_up"]:
                intent = "inventory_check"
                print(f"[OVERRIDE] Forcing intent to inventory_check for selection: '{user_message}'")

    # Fallback: treat direct shopping requests as stock checks when intent falls back to general chat.
    if intent == "general_chat":
        normalized = user_message.lower()
        stock_keywords = re.compile(r"\b(i need|need|i want|want|looking for|looking to buy|find|got any|have any|have|do you have)\b")
        recipe_keywords = re.compile(r"\b(recipe|cook|make|how to|how do i|what can i make|suggest|budget|under \$|under \d+)\b")
        if stock_keywords.search(normalized) and not recipe_keywords.search(normalized):
            intent = "stock_check"
        else:
            words = re.findall(r"\w+", normalized)
            ignore_short = {"yes", "y", "yep", "yeah", "no", "nope", "nah", "thanks", "thank", "thank you", "ok", "okay", "sure", "please", "hi", "hello", "bye", "goodbye"}
            if 0 < len(words) <= 3 and not recipe_keywords.search(normalized) and any(w not in ignore_short for w in words):
                intent = "stock_check"
    llm_response = ""

    # Follow up detection overide
    state = get_user_state(user_id)
    last_intent = state.get("last_intent")
    last_response = state.get("last_response")

    # If the last intent was inventory check and the user message is short likely a selection 
    if last_intent in ["inventory_check", "stock_check", "recommend_recipe"]:
        print(f"Check 1: [OVERRIDE] Changing intent from {intent} to follow_up based on conversation context")
        words = user_message.lower().split()
        # Short message (1-3 words) that's not a greeting
        if 1 <= len(words) <= 3 and not any(word in words for word in ["hi", "hello", "hey", "thanks"]):
            intent = "follow_up"
            print(f"Check 2: [OVERRIDE] Changing intent from {intent} to follow_up based on conversation context")

    # Ensure the FINAL intent is saved so llm_route.py doesn't overwrite it with the raw LLM intent
    state["last_intent"] = intent

    match intent:
        case "stock_check":
            from services.llm_logic import search_product_with_fallback
            
            # Entities have already been extracted by llm_response_extraction
            product_name = entities.product_name if entities.product_name else ""
            
            # Use the new fallback handler
            dict_response = await search_product_with_fallback(user_id, product_name, entities, user_message)
            
            print(f"Fallback response for stock check: {dict_response}")
            return json.dumps(dict_response)

        case "get_recipe":
            # Check if the intent is "get_recipe"
            #Check recipe availability
            system_prompt = await prepare_get_recipe_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)
            #print("LLM response:", llm_response)

            return llm_response

        case "add_to_cart":
            from services.llm_response import get_products
            from intents.cart_logic import get_cart_items

            session_products = get_products(user_id)
            msg_lower = user_message.lower().strip()

            # -------------------------------
            # 1. HARD MATCHING (NO LLM)
            # -------------------------------
            if session_products:

                user_tokens = set(re.findall(r'\w+', msg_lower)) - {
                    "add", "the", "to", "cart", "please", "my", "some", "a", "an", "i", "want"
                }

                best_match = None
                best_score = 0

                for p in session_products:
                    p_name = p["product_name"].lower()
                    p_tokens = set(re.findall(r'\w+', p_name))

                    overlap = len(user_tokens.intersection(p_tokens))
                    substring = any(token in p_name for token in user_tokens if len(token) > 2)

                    score = overlap * 2
                    if substring:
                        score += 2
                    if p_name in msg_lower:
                        score += 5

                    if score > best_score:
                        best_score = score
                        best_match = p

                # -------------------------------
                # 2. IF WE FOUND A MATCH → RETURN DIRECTLY
                # -------------------------------
                if best_match:
                    print("✅ PYTHON MATCH:", best_match["product_name"])

                    return json.dumps({
                        "message": f"Perfect! I've added {best_match['product_name']} to your cart.",
                        "added_products": [
                            {
                                "product_name": best_match["product_name"],
                                "quantity": 1,
                                "price": best_match["price"]
                            }
                        ],
                        "action_ready": True
                    })

                # -------------------------------
                # 3. AFFIRMATION → PICK FIRST
                # -------------------------------
                affirmations = {"yes", "yeah", "yep", "ok", "okay", "sure", "add it", "that one"}

                if msg_lower in affirmations:
                    first = session_products[0]

                    print("✅ DEFAULT PICK:", first["product_name"])

                    return json.dumps({
                        "message": f"Great choice! I've added {first['product_name']} to your cart.",
                        "added_products": [
                            {
                                "product_name": first["product_name"],
                                "quantity": 1,
                                "price": first["price"]
                            }
                        ],
                        "action_ready": True
                    })

            # -------------------------------
            # 4. FALLBACK → USE LLM ONLY IF NEEDED
            # -------------------------------
            print("⚠️ Falling back to LLM for add_to_cart")

            system_prompt = await prepare_cart_response(entities, user_message, user_id)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "remove_from_cart":
            from services.llm_logic import prepare_remove_from_cart_response
            system_prompt = await prepare_remove_from_cart_response(entities, user_message, user_id)
            llm_response = await message_to_llm(system_prompt, " ", chat_history)

            return llm_response
        
        case "view_cart":
            from intents.cart_logic import formatted_cart_summary
            cart_message = formatted_cart_summary(user_id)
            return json.dumps({
                "message": cart_message,
                "action_ready": False
            })
        case "check_recipe_availability":
            # Handle recipe availability
            system_prompt = await prepare_check_recipe_availability_response(entities, user_message)
            llm_response  = await message_to_llm(system_prompt, " ", chat_history)

            #print("Recipe LLM response:", llm_response)
            return llm_response
        
        case "recommend_recipe":
            # Check if user just checked their cart and asking 
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
            return json.dumps({"message": "Sure! Heres your cart link when youre ready to checkout. \n https://localhost:3000/cart",
                               'action': 'redirect_to_cart',
                               'redirect_url': '/cart'})

        case "terms_and_conditions":
            # Use RAG to fetch the terms and conditions from a local file or database
            from intents.terms_and_conditions import terms_and_conditions, prepare_terms_and_conditions_response
            system_prompt = await prepare_terms_and_conditions_response(user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response

        case "checkout":            
            return json.dumps({"message": "Sure! Heres your checkout link when youre ready to checkout. \n https://localhost:3000/checkout",
                               'action': 'redirect_to_checkout',
                               'redirect_url': '/checkout'})
        
        case "product_location":
            from services.llm_logic import prepare_aisle_location_response
            system_prompt = await prepare_aisle_location_response(entities, user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response
        
        case "store_info":
            from intents.terms_and_conditions import prepare_store_info_response
            system_prompt = await prepare_store_info_response(user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response
        
        case "inventory_check":
            from intents.stock_check import (
                is_subcategory_selection, 
                get_selected_subcategory,
                is_product_selection,
                get_selected_products,
                get_products_by_subcategory,
                get_subcategories_in_category
            )
            from services.context_handler import is_confirmation_response as is_confirm
            from services.llm_logic import prepare_add_inventory_to_cart_response
            from services.llm_response import get_products as get_session_products
            
            state = get_user_state(user_id)
            
            print(f"[INVENTORY] Processing: '{user_message}'")
            print(f"[INVENTORY] last_intent: {state.get('last_intent')}")
            print(f"[INVENTORY] has subcategories: {bool(state.get('last_subcategories'))}")
            print(f"[INVENTORY] has products: {bool(state.get('last_inventory_products'))}")

            # Fall back llm search category prompt
            fall_back_prompt = f"""
                    You are a strict classification system for a supermarket inventory.

                    Your job is to map a user's query to ONE category.

                    You already have a keyword system. Use similar reasoning, but handle:
                    - vague language
                    - misspellings
                    - informal speech (including Patois)

                    USER INPUT:
                    {user_message}

                    VALID CATEGORIES (choose ONLY one):
                    - International Foods
                    - Condiments & Sauces
                    - Beverages
                    - Baking Supplies
                    - Vinegar
                    - Canned & Jarred
                    - Grains & Rice
                    - Snacks
                    - Sugar & Sweeteners
                    - Produce
                    - Fruits
                    - Bakery
                    - Spices & Seasonings
                    - Household
                    - Meat & Poultry
                    - Frozen Foods
                    - Dry Goods
                    - Dairy & Eggs
                    - Breakfast
                    - Nuts & Seeds
                    - Prepared Foods
                    - Seafood
                    - Health Foods
                    - Pet Supplies
                    - Meal Kits
                    - Tobacco
                    - Specialty
                    - Other

                    RULES:
                    - Return ONLY one category from the list
                    - If unsure, return "Other"
                    - Do NOT explain anything
                    - Do NOT return anything except JSON

                    OUTPUT:
                    {{
                        "category": "<one category from the list>"
                    }}
                    """

            # STEP 1 & 2: LLM-based selection for Products or Subcategories
            last_products = state.get("last_inventory_products", [])
            last_subs = state.get("last_subcategories", [])
            
            selection_made = None
            
            if last_products:
                # Use LLM to extract product selection
                from services.llm_logic import prepare_inventory_selection_response
                options = [p['product_name'] for p in last_products]
                print("Building product selection prompt for LLM...")
                prompt = await prepare_inventory_selection_response(user_message, options)
                print("Checking product selection with LLM...")
                res_raw = await message_to_llm(prompt, user_message, chat_history)

                print(f"[PROMPT]: {prompt}")
                print(f"[RESPONSE]: {res_raw}")
            

                try:
                    res_json = json.loads(res_raw)
                    selection_made = res_json.get("selected_inventory")
                except:
                    selection_made = None
                
                if selection_made and selection_made != "null":
                    selected_products = [selection_made]
                    
                    # Check if user explicitly wants to add to cart
                    add_keywords = ["add", "put", "buy", "cart", "purchase", "get", "want", "ill take", "ill get"]
                    wants_to_add = any(kw in user_message.lower() for kw in add_keywords)
                    
                    if wants_to_add:
                        # User said "add oxtail to cart" or similar
                        product_details = []
                        for name in selected_products:
                            product = next((p for p in state["last_inventory_products"] if p['product_name'] == name), None)
                            if product:
                                product_details.append({
                                    "product_name": product['product_name'],
                                    "quantity": 1,
                                    "price": product['price']
                                })
                        
                        if product_details:
                            return json.dumps({
                                "message": f"Perfect! I've added {', '.join(selected_products)} to your cart.",
                                "added_products": product_details,
                                "action_ready": True
                            })
                    else:
                        # User just selected a product, ask if they want to add it
                        product_info = ""
                        for name in selected_products:
                            product = next((p for p in state["last_inventory_products"] if p['product_name'] == name), None)
                            if product:
                                stock_status = "✅ In Stock" if product.get('in_stock', True) else "❌ Out of Stock"
                                product_info += f"\n• {product['product_name']} — ${product['price']} ({stock_status})"
                        
                        return json.dumps({
                            "message": f"Great choice! {selected_products[0]} is ${next((p['price'] for p in state['last_inventory_products'] if p['product_name'] == selected_products[0]), 'available')}. Would you like me to add it to your cart?",
                            "product_names": selected_products,
                            "action_ready": False
                        })

            # Subcategory selection
            if is_subcategory_selection(user_message, user_id):
                selected_subcategory = get_selected_subcategory(user_message)
                print(f"[INVENTORY] Subcategory selection detected (user): '{selected_subcategory}'")
                selection_made = selected_subcategory

            
            # Subcategory selection
            if not selection_made and last_subs:
                from services.llm_logic import prepare_inventory_selection_response
                options = [s['subcategory'] for s in last_subs]
                prompt = await prepare_inventory_selection_response(user_message, options)
                res_raw = await message_to_llm(prompt, user_message, chat_history)
                
                try:
                    res_json = json.loads(res_raw)
                    selection_made = res_json.get("selected_inventory")
                except:
                    selection_made = None
                    
                if selection_made and selection_made != "null":
                    selected_subcategory = selection_made
                    last_category = state.get("last_category")
                    
                    print(f"[INVENTORY] Subcategory selection detected (LLM): '{selected_subcategory}' in category '{last_category}'")
                    
                    if last_category:
                        # Get all products in this subcategory
                        products = get_products_by_subcategory(last_category, selected_subcategory)
                        
                        if products:
                            # Store products in state for next turn
                            state["last_inventory_products"] = products
                            
                            # SAVE PRODUCTS to the session pool so add_to_cart logic can find them
                            from services.llm_response import save_products
                            save_products(user_id, products)
                            
                            # Build product list for LLM response
                            product_list = ""
                            for p in products[:10]:  # Limit to first 10 for display
                                stock = "✅" if p.get('in_stock', True) else "❌"
                                product_list += f"\n• {p['product_name']} — ${p['price']} {stock}"
                            
                            if len(products) > 10:
                                product_list += f"\n• ... and {len(products) - 10} more items"
                            
                            # Create system prompt for LLM to format response
                            system_prompt = f"""
                            You are a helpful UCC Supermarket assistant.
                            User asked about {selected_subcategory} in {last_category}.
                            
                            Available products:
                            {product_list}
                            
                            INSTRUCTIONS:
                            - List the products conversationally (NOT as bullet points in your response)
                            - Mention prices
                            - If there are many products, summarize the range
                            - Ask which specific product they want or suggest "the first one", "the second one"
                            - Be friendly and helpful
                            
                            OUTPUT (JSON only):
                            {{
                                "message": "Your conversational response here",
                                "subcategory": "{selected_subcategory}",
                                "products_found": {len(products)},
                                "products": {json.dumps(products[:10])},
                                "action_ready": false
                            }}
                            """
                            
                            llm_response = await message_to_llm(system_prompt, " ", chat_history)
                            return llm_response
                        else:
                            # No products found in this subcategory
                            return json.dumps({
                                "message": f"I'm sorry, we don't have any {selected_subcategory} in stock right now. Would you like to see something else from {last_category}?",
                                "action_ready": False
                            })
            
            # STEP 3: Check if user wants to ADD a previously selected product (affirmation case)
            elif state.get("pending_selection") and is_confirm(user_message) == "yes":
                selected_products = state.get("pending_selection", [])
                product_details = []
                
                for name in selected_products:
                    # Try to get product details from last_inventory_products or from inventory
                    product = None
                    if state.get("last_inventory_products"):
                        product = next((p for p in state["last_inventory_products"] if p['product_name'] == name), None)
                    
                    if product:
                        product_details.append({
                            "product_name": product['product_name'],
                            "quantity": 1,
                            "price": product['price']
                        })
                    else:
                        # Fallback: just add name without price
                        product_details.append({
                            "product_name": name,
                            "quantity": 1,
                            "price": 0
                        })
                
                if product_details:
                    # Clear pending selection
                    state["pending_selection"] = []
                    # Clear last inventory products to avoid confusion next turn
                    state["last_inventory_products"] = []
                    
                    return json.dumps({
                        "message": f"Great! I've added {', '.join(selected_products)} to your cart.",
                        "added_products": product_details,
                        "action_ready": True
                    })
            
            # STEP 4: User wants to see SUBCATEGORIES (initial inventory check)
            else:
                # Extract category from user message or use existing category context
                from intents.stock_check import extract_category_from_query, get_subcategories_in_category
                
                category, success = extract_category_from_query(user_message)

                print(f"Initial inventory check. Detected category: '{category}'")

                
                # If no category detected, ask user to specify
                if category:
                    # Get subcategories for this category
                    subcategories = get_subcategories_in_category(category)
                    
                    if not subcategories:
                        return json.dumps({
                            "message": f"Sorry, we don't currently have any items in {category}. Would you like to browse another category?",
                            "action_ready": False
                        })
                    
                    # Store subcategories for next turn
                    state["last_subcategories"] = subcategories
                    state["last_category"] = category
                    state["last_inventory_products"] = []  # Clear any previous products
                    state["pending_selection"] = []  # Clear any pending selections
                    
                    # Build subcategory list for LLM
                    subcategory_list = ""
                    for i, sub in enumerate(subcategories[:15], 1):
                        subcategory_list += f"\n• {sub['subcategory']} ({sub['count']} items)"
                    
                    if len(subcategories) > 15:
                        subcategory_list += f"\n• ... and {len(subcategories) - 15} more types"
                    
                    # Create system prompt for LLM to format response
                    system_prompt = f"""
                    You are a helpful UCC Supermarket assistant.
                    User asked: "{user_message}"
                    
                    Available subcategories in {category}:
                    {subcategory_list}
                    
                    INSTRUCTIONS:
                    - List the types conversationally (NOT as bullet points in your response)
                    - Make it feel natural, not robotic
                    - Ask which type they want to explore
                    - Suggest they can say the name or "the first one", "the second one"
                    - Be friendly and helpful
                    
                    OUTPUT (JSON only):
                    {{
                        "message": "Your conversational response here",
                        "category": "{category}",
                        "subcategories_count": {len(subcategories)},
                        "action_ready": false
                    }}
                    """
                    
                    llm_response = await message_to_llm(system_prompt, " ", chat_history)
                    return llm_response
                elif success == False:

                    category_response = await message_to_llm(fall_back_prompt, user_message, chat_history)

                    # Extract category from LLM response
                    category = json.loads(category_response).get("category")
                    print(f"LLM fallback category detection: '{category}'")

                    if category == "Other":
                        return json.dumps({
                            "message": "I'm sorry, I couldn't determine a specific category from your request."
                            " Could you please specify if you're looking for something like 'fruits', 'snacks', 'beverages', etc.?",
                            "action_ready": False
                        })
                    
                    print(f"Initial inventory check. Detected category: '{category}'")

                
                    # If no category detected, ask user to specify
                    if category:

                        # Get subcategories for this category
                        subcategories = get_subcategories_in_category(category)
                        
                        if not subcategories:
                            return json.dumps({
                                "message": f"Sorry, we don't currently have any items in {category}. Would you like to browse another category?",
                                "action_ready": False
                            })
                        
                        # Store subcategories for next turn
                        state["last_subcategories"] = subcategories
                        state["last_category"] = category
                        state["last_inventory_products"] = []  # Clear any previous products
                        state["pending_selection"] = []  # Clear any pending selections
                        
                        # Build subcategory list for LLM
                        subcategory_list = ""
                        for i, sub in enumerate(subcategories[:15], 1):
                            subcategory_list += f"\n• {sub['subcategory']} ({sub['count']} items)"
                        
                        if len(subcategories) > 15:
                            subcategory_list += f"\n• ... and {len(subcategories) - 15} more types"
                        

                        # Create system prompt for LLM to format response
                        system_prompt = f"""
                        You are a helpful UCC Supermarket assistant.
                        User asked: "{user_message}"
                        
                        Available subcategories in {category}:
                        {subcategory_list}
                        
                        INSTRUCTIONS:
                        - List the types conversationally (NOT as bullet points in your response)
                        - Make it feel natural, not robotic
                        - Ask which type they want to explore
                        - Suggest they can say the name or "the first one", "the second one"
                        - Be friendly and helpful
                        
                        OUTPUT (JSON only):
                        {{
                            "message": "Your conversational response here",
                            "category": "{category}",
                            "subcategories_count": {len(subcategories)},
                            "action_ready": false
                        }}
                        """
                        # Update user state with the newly detected category even if it came from LLM fallback
                        state["last_category"] = category

                    llm_response = await message_to_llm(system_prompt, " ", chat_history)
                    return llm_response
                    
                else:
                    return json.dumps({
                        "message": "I'm sorry, It seems like we dont have that category in our store."
                        " You can ask about a specific product in that category or choose another category to explore. (e.g., 'Do you have any fresh",
                        "action_ready": False
                    })
                
        case "general_chat":
            from intents.terms_and_conditions import prepare_general_chat_response
            system_prompt = await prepare_general_chat_response(user_message)
            llm_response = await message_to_llm(system_prompt, user_message, chat_history)

            return llm_response

        case "follow_up":
            from intents.stock_check import (
                get_subcategories_in_category,
                get_products_by_subcategory,
                is_subcategory_selection,
                get_selected_subcategory,
                is_product_selection,
                get_selected_products,
                check_and_update_product_selection
            )
            from services.context_handler import is_confirmation_response as is_confirm
            
            state = get_user_state(user_id)
            last_intent = state.get("last_intent")
            last_subcategories = state.get("last_subcategories", [])
            last_products = state.get("last_inventory_products", [])
            last_category = state.get("last_category")
            recipe_name = state.get("last_recipe_name")
            last_suggestions = state.get("last_suggestions")

            
            print(f"[FOLLOW_UP] last_intent: {last_intent}")
            print(f"[FOLLOW_UP] has subcategories: {bool(last_subcategories)}")
            print(f"[FOLLOW_UP] has products: {bool(last_products)}")
            print(f"[FOLLOW_UP] last_category: {last_category}")
            print(f"[FOLLOW_UP] last_recipe_name: {recipe_name}")
            
            # CASE 1: User is selecting from SUBCATEGORIES (not products yet)
            if last_subcategories and not last_products:
                print("[FOLLOW_UP] CASE 1: Subcategory selection detected")
                print(f"[FOLLOW_UP] Available subcategories: {[s['subcategory'] for s in last_subcategories]}")
                from intents.stock_check import extract_subcategory_from_response, get_products_by_subcategory
                from services.llm_response import save_products
                
                # ========== LAYER 1: TRY FAST MATCHING FIRST ==========
                available_subcategories = [s['subcategory'] for s in last_subcategories]
                user_subcategory_selection = extract_subcategory_from_response(user_message, available_subcategories)
            
                if user_subcategory_selection:
                    print(f"[FOLLOW_UP] Fast match found: {user_subcategory_selection}")
                    
                    # Search for products in this subcategory
                    user_product_selection = get_products_by_subcategory(last_category, user_subcategory_selection)

                    print(f"[FOLLOW_UP] Products found for '{user_subcategory_selection}': {[p['product_name'] for p in user_product_selection]}")

                    if user_product_selection:
                        state["last_inventory_products"] = user_product_selection
                        save_products(user_id, user_product_selection)
            
                        # Build response showing products
                        product_list = ""
                        for p in user_product_selection[:10]:
                            stock = "✅" if p.get('in_stock', True) else "❌"
                            product_list += f"\n• {p['product_name']} — ${p['price']} {stock}"
                        
                        
                        # List the products to the user
                        return json.dumps({
                            "message": f"Here are the products we have in {user_subcategory_selection}:\n{product_list}\nWhich one would you like to explore?",
                            "products": user_product_selection,
                            "action_ready": True
                        })  
                    
                    else:
                        # Subcategory matched but no products found
                        return json.dumps({
                            "message": f"Sorry, we don't have any items in {user_subcategory_selection} right now.",
                            "action_ready": False
                        })
            
                # ========== LAYER 2: FALL BACK TO LLM IF FAST MATCH FAILED ==========
                print("[FOLLOW_UP] Fast match failed, using LLM fallback")
                from services.llm_logic import prepare_inventory_selection_response
                
                options = [s['subcategory'] for s in last_subcategories]
                prompt = await prepare_inventory_selection_response(user_message, options)
                res_raw = await message_to_llm(prompt, user_message, chat_history)
                
                try:
                    res_json = json.loads(res_raw)
                    selection_made = res_json.get("selected_inventory")
                except:
                    selection_made = None
                
                if selection_made and selection_made != "null":
                    # ✅ LLM found a match → fetch its products
                    print(f"[FOLLOW_UP] LLM match found: {selection_made}")
                    
                    if last_category:
                        products = get_products_by_subcategory(last_category, selection_made)
                        
                        if products:
                            state["last_inventory_products"] = products
                            save_products(user_id, products)
                            
                            # Build response showing products
                            product_list = ""
                            for p in products[:10]:
                                stock = "✅" if p.get('in_stock', True) else "❌"
                                product_list += f"\n• {p['product_name']} — ${p['price']} {stock}"
                            
                            system_prompt = f"""
                            You are a helpful UCC Supermarket assistant.
                            User selected "{selection_made}" from {last_category}.
                            
                            Available products in {selection_made}:
                            {product_list}
                            
                            INSTRUCTIONS:
                            - List products conversationally
                            - Mention prices and stock status
                            - Ask which product they'd like or suggest "the first one", "the second one"
                            - Be friendly and helpful
                            
                            OUTPUT (JSON only):
                            {{
                                "message": "Your response here",
                                "subcategory": "{selection_made}",
                                "products_found": {len(products)},
                                "action_ready": false
                            }}
                            """
                            
                            llm_response = await message_to_llm(system_prompt, " ", chat_history)
                            return llm_response  # ✅ EARLY RETURN
                        
                        else:
                            # Subcategory matched but no products
                            return json.dumps({
                                "message": f"Sorry, we don't have any items in {selection_made} right now.",
                                "action_ready": False
                            })
            
                # ========== LAYER 3: NOTHING MATCHED - ASK FOR CLARIFICATION ==========
                print("[FOLLOW_UP] No match found, asking for clarification")
                return json.dumps({
                    "message": f"I didn't catch that. Could you pick from: {', '.join([s['subcategory'] for s in last_subcategories[:5]])}?",
                    "action_ready": False
                })
 


            # ✅ CASE 2: User is selecting from PRODUCTS (after subcategory was selected)
            elif last_products:
                print("[FOLLOW_UP] CASE 2: Product selection detected")
                from services.llm_logic import prepare_inventory_selection_response
                
                options = [p['product_name'] for p in last_products]
                prompt = await prepare_inventory_selection_response(user_message, options)
                res_raw = await message_to_llm(prompt, user_message, chat_history)
                
                try:
                    res_json = json.loads(res_raw)
                    selection_made = res_json.get("selected_inventory")
                except:
                    selection_made = None
                
                if selection_made and selection_made != "null":
                    # Product matched
                    state["pending_selection"] = [selection_made]
                    product = next((p for p in last_products if p['product_name'] == selection_made), None)
                    
                    if product:
                        return json.dumps({
                            "message": f"Great choice! {product['product_name']} is ${product['price']}. Would you like me to add it to your cart?",
                            "product_names": [selection_made],
                            "action_ready": False
                        })
                
                # Check for "yes" confirmation
                if is_confirm(user_message) == "yes" and state.get("pending_selection"):
                    selected = state.get("pending_selection", [])
                    state["pending_selection"] = []
                    
                    product_details = []
                    for name in selected:
                        product = next((p for p in last_products if p['product_name'] == name), None)
                        if product:
                            product_details.append({
                                "product_name": product['product_name'],
                                "quantity": 1,
                                "price": product['price']
                            })
                    
                    return json.dumps({
                        "message": f"Perfect! I've added {', '.join(selected)} to your cart.",
                        "added_products": product_details,
                        "action_ready": True
                    })
                
                # No match for product
                return json.dumps({
                    "message": f"I didn't catch that. Could you pick from: {', '.join([p['product_name'] for p in last_products[:3]])}?",
                    "action_ready": False
                })
            


            # CASE C: Following up on recipe recommendations
            elif last_intent in ["recommend_recipe", "recipe_from_cart_items", "get_recipe"] and last_suggestions:
                print("[FOLLOW_UP] CASE A: Checking for recipe selection...")
                
                # Use the helper function to detect recipe selection
                if is_recipe_selection(user_message, last_suggestions):
                    print(f"[FOLLOW_UP] Recipe selection detected!")
                    
                    from services.context_handler import fuzzy_match
                    
                    text = user_message.lower().strip()
                    
                    # Try to find the selected recipe
                    selected_recipe = None
                    
                    # Method 1: Fuzzy match against suggestions
                    for recipe in last_suggestions:
                        if fuzzy_match(text, recipe):
                            selected_recipe = recipe
                            print(f"[FOLLOW_UP] Fuzzy match found: {recipe}")
                            break
                    
                    # Method 2: Ordinal selection (first, second, etc.)
                    if not selected_recipe:
                        ordinals = {
                            "first": 0, "1st": 0, "one": 0,
                            "second": 1, "2nd": 1, "two": 1,
                            "third": 2, "3rd": 2, "three": 2,
                        }
                        for ordinal, index in ordinals.items():
                            if ordinal in text and index < len(last_suggestions):
                                selected_recipe = last_suggestions[index]
                                print(f"[FOLLOW_UP] Ordinal match found: {ordinal} → {selected_recipe}")
                                break
                    
                    if selected_recipe:
                        # User selected a recipe - get the full recipe
                        print(f"[FOLLOW_UP] Getting full recipe for: {selected_recipe}")
                        
                        state["selected_recipe"] = selected_recipe
                        
                        # Call prepare_get_recipe_response to format the prompt                        
                        system_prompt = await prepare_get_recipe_response(
                            Entity(recipe_name=selected_recipe),
                            f"Give me the full recipe for {selected_recipe}"
                        )
                        llm_response = await message_to_llm(system_prompt, " ", chat_history)
                        return llm_response
                    
                    # If we detected recipe selection intent but couldn't find a match
                    print("[FOLLOW_UP] Recipe selection detected but no recipe matched")
                    return json.dumps({
                        "message": f"I didn't quite catch which recipe you want. Could you pick from: {', '.join(last_suggestions[:3])}?",
                        "action_ready": False
                    })

            # CASE E: General follow-up - let LLM handle with context
            else:
                # Pass to general chat with full context
                from intents.terms_and_conditions import prepare_general_chat_response
                print("[FOLLOW_UP] CASE E: No specific follow-up detected, passing to general chat with context")
                system_prompt = await prepare_general_chat_response(user_message)
                
                # Include the last few messages for context
                context_messages = chat_history[-4:] if len(chat_history) > 4 else chat_history
                context_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context_messages])
                
                enhanced_prompt = f"""
                Previous conversation:
                {context_str}
                
                User now says: "{user_message}"
                
                {system_prompt}
                """
                
                llm_response = await message_to_llm(enhanced_prompt, user_message, chat_history)
                return llm_response

    return llm_response