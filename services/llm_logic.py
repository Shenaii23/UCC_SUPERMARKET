# services/llm_logic.py
import json

import json

from models.llm_classes import Entity

# intent files
from intents.check_recipe_availability import get_recipe_data
from services.context_handler import update_user_state, get_user_state
from intents.stock_check import (get_proactive_matches, extract_category_from_query)
from datasets.data import inventory_df
from typing import List, Dict


from typing import Tuple, Optional, List, Dict
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)

async def prepare_stock_response(entities: Entity, user_message: str):
    """
    Search for products in inventory and return results.
    
    Returns:
        Tuple[Union[Dict, List[Dict], None], bool]:
            - (product/products, True) if found
            - (None, False) if not found (with fallback prompt prepared)
    
    Examples:
        User: "do you have milk"
        → ({"product_name": "Whole Milk", ...}, True)
        
        User: "do you have xyz123"
        → (None, False)  # Fallback: "We couldn't find xyz123. Did you mean...?"
    """
    
    product_to_find = entities.product_name.lower().strip()
    
    # Handle plural forms
    if product_to_find.endswith('s') and not product_to_find.endswith('ss'):
        singular_version = product_to_find[:-1]
        product_to_find = f"{product_to_find}|{singular_version}"
        print(f"DEBUG: Searching for both plural and singular: '{product_to_find}'")
    
    # Get category hint
    category = entities.category_hint
    category = extract_category_from_query(category)
    
    # Search inventory
    results = get_proactive_matches(product_to_find, inventory_df, threshold=65)
    
    # ========== CASE 1: PRODUCTS FOUND ==========
    if results:
        logger.info(f"[STOCK] Found {len(results)} products for '{product_to_find}'")
        
        # Format context for LLM
        context = "\n".join([
            f"- {r.get('product_name', '')} (Quantity: {r.get('quantity', 0)}, Price: ${r.get('price', 0)})"
            for r in results
        ])
        
        inventory_prompt = """
        You are a helpful grocery store assistant.

        The users original message is: "{user_message}"
        The user searched for: "{product_to_find}"

        Here are the matching products from our inventory:
        {context}

        YOUR TASK:
        1. Select products that match exactly what the user asked for.
        2. When in doubt, INCLUDE the product - let the user decide
        3. Only exclude products that are clearly unrelated
        4. Include all variants and related products
        5. Provide a friendly, natural response using PLAIN TEXT with newlines
        6. List products with each on a new line
        7. At the end, ask if they'd like to add any to their cart
        8. Use only plain text formatting - NO HTML tags

        OUTPUT FORMAT (JSON only):
        {{
            "message": "A friendly message to show the user",
            "products": [
                {{
                    "product_name": "exact name from inventory",
                    "quantity": available quantity,
                    "price": price
                }}
            ],
            "action_ready": true
        }}
        """.format(user_message=user_message, product_to_find=product_to_find, context=context)
        
        try:
            from services.user_intent_executive import message_to_llm
            llm_response = await message_to_llm(inventory_prompt, user_message, [])
            parsed = json.loads(llm_response)
            return (parsed.get("products", results), True)
        except:
            # If LLM response parsing fails, return raw results
            return (results, True)
    
    # ========== CASE 2: NO PRODUCTS FOUND ==========
    else:
        logger.info(f"[STOCK] No products found for '{product_to_find}'")
        
        # ✅ RETURN: (None, False) - Triggers fallback
        return (None, False)


# ========== HELPER: Get product with fallback handling ==========
async def search_product_with_fallback(
    user_id: str,
    product_name: str,
    entities: Entity,
    user_message: str = ""
) -> Dict:
    """
    Search for a product and handle fallback if not found.
    
    Returns:
        {
            "found": bool,
            "products": [...] or None,
            "message": str,
            "action_ready": bool,
            "fallback_prompted": bool
        }
    
    Examples:
        Product found:
        {
            "found": True,
            "products": [...],
            "message": "We have milk: Whole Milk ($350), Skim Milk ($300)",
            "action_ready": True,
            "fallback_prompted": False
        }
        
        Product not found:
        {
            "found": False,
            "products": None,
            "message": "We couldn't find xyz123. Did you mean apple juice?",
            "action_ready": False,
            "fallback_prompted": True
        }
    """
    
    # Use passed entities
    entities.product_name = product_name
    
    # Search inventory
    products, found = await prepare_stock_response(entities, user_message)
    
    # ========== HANDLE FOUND CASE ==========
    if found and products:
        logger.info(f"[SEARCH] Found products: {len(products) if isinstance(products, list) else 1}")
        
        return {
            "found": True,
            "products": products if isinstance(products, list) else [products],
            "message": format_product_message(products),
            "action_ready": True,
            "fallback_prompted": False
        }
    
    # ========== HANDLE NOT FOUND CASE - TRIGGER FALLBACK ==========
    else:
        
        # Generate helpful fallback message
        fallback_message = await generate_fallback_prompt(
            product_name=product_name,
            user_message=user_message
        )
        
        return {
            "found": False,
            "products": None,
            "message": fallback_message,
            "action_ready": False,
            "fallback_prompted": True
        }


# ========== HELPER: Generate fallback prompt when product not found ==========
async def generate_fallback_prompt(product_name: str, user_message: str = "") -> str:
    """
    Generate a helpful fallback message when product is not found.
    
    Includes:
    1. Empathetic message
    2. Suggestions for similar products (if available)
    3. Help text (spelling, search tips)
    
    Returns:
        str: User-friendly message
    """
    
    # Try to find similar products for suggestions
    similar_products = find_similar_products(product_name, max_results=3)
    
    fallback_prompt = f"""
    You are a helpful UCC Supermarket assistant.
    
    The user searched for: "{product_name}"
    Full message: "{user_message}"
    
    Unfortunately, we don't currently have "{product_name}" in stock.
    
    Here are some similar products we do have:
    {format_similar_products(similar_products)}
    
    YOUR TASK:
    1. Acknowledge that we don't have the product they searched for
    2. Suggest the similar products we found (if any)
    3. Offer helpful alternatives (e.g., "We have fresh apples, apple juice, or apple sauce")
    4. Give them suggestions on what to try next
    5. Be friendly and helpful, not apologetic
    6. Use plain text formatting only
    
    EXAMPLES:
    
    User searched: "xyz123"
    → "We don't currently have xyz123 in stock. Did you mean apple juice? 
         We have several options like Fresh Apple, Apple Juice, or Apple Sauce. 
         Would you like to search for any of these?"
    
    User searched: "rare ingredient"
    → "We don't have rare ingredient in stock right now. However, we have similar products 
         like Ingredient Option 1, Ingredient Option 2, or Ingredient Option 3. 
         Feel free to try searching for one of these, or let me know what else you need!"
    
    OUTPUT (JSON only):
    {{
        "message": "Your helpful fallback message here"
    }}
    """
    
    try:
        from services.user_intent_executive import message_to_llm
        response = await message_to_llm(fallback_prompt, product_name, [])
        parsed = json.loads(response)
        return parsed.get("message", get_default_fallback_message(product_name))
    except Exception as e:
        logger.error(f"[FALLBACK] Error generating fallback: {e}")
        return get_default_fallback_message(product_name)


# ========== HELPER: Default fallback message ==========
def get_default_fallback_message(product_name: str) -> str:
    """
    Return a default fallback message when LLM generation fails.
    
    Safe fallback that always works.
    """
    
    similar = find_similar_products(product_name, max_results=2)
    
    base_message = f"We don't currently have {product_name} in stock."
    
    if similar:
        suggestions = ", ".join([p["product_name"] for p in similar])
        return f"{base_message} Did you mean one of these? {suggestions}. Would you like to add any of these to your cart instead?"
    
    return (
        f"{base_message} "
        f"Would you like to search for something else, or would you like me to suggest some popular items?"
    )


# ========== HELPER: Find similar products ==========
def find_similar_products(
    search_term: str,
    max_results: int = 3,
    threshold: int = 60
) -> List[Dict]:
    """
    Find products similar to the search term for fallback suggestions.
    
    Uses fuzzy matching to find related products.
    
    Args:
        search_term: Product to search for
        max_results: Max similar products to return
        threshold: Minimum fuzzy match score (0-100)
    
    Returns:
        List of similar products with match scores
    """
    
    search_lower = search_term.lower()
    similar = []
    
    for _, row in inventory_df.iterrows():
        product_name = row.get("product_name", "").lower()
        
        # Skip exact match (we already know this wasn't found)
        if product_name == search_lower:
            continue
        
        # Calculate similarity score
        fuzzy_score = fuzz.ratio(search_lower, product_name)
        partial_score = fuzz.partial_ratio(search_lower, product_name)
        combined_score = (fuzzy_score * 0.6) + (partial_score * 0.4)
        
        # Include if above threshold
        if combined_score >= threshold:
            similar.append({
                "product_name": row.get("product_name"),
                "match_score": round(combined_score, 1),
                "quantity": row.get("quantity", 0),
                "price": row.get("price", 0)
            })
    
    # Sort by match score and return top N
    similar.sort(key=lambda x: x["match_score"], reverse=True)
    return similar[:max_results]


# ========== HELPER: Format similar products for prompt ==========
def format_similar_products(products: List[Dict]) -> str:
    """
    Format similar products for display in fallback message.
    """
    
    if not products:
        return "(No similar products found)"
    
    formatted = []
    for i, p in enumerate(products, 1):
        name = p.get("product_name", "Unknown")
        score = p.get("match_score", 0)
        formatted.append(f"• {name} (Match: {score}%)")
    
    return "\n".join(formatted)


# ========== HELPER: Format product message ==========
def format_product_message(products: Union[Dict, List[Dict]]) -> str:
    """
    Format products for user display.
    """
    
    if not products:
        return "No products available"
    
    # Handle single product
    if isinstance(products, dict):
        products = [products]
    
    if len(products) == 1:
        p = products[0]
        return f"We have {p.get('product_name')} in stock (${p.get('price')}). Would you like to add it to your cart?"
    
    # Handle multiple products
    product_list = []
    for p in products:
        product_list.append(f"• {p.get('product_name')} (Quantity: {p.get('quantity')}, Price: ${p.get('price')})")
    
    return f"We have several options for you:\n" + "\n".join(product_list) + "\n\nWould you like to add any of these to your cart?"

async def prepare_cart_response(entities: Entity, user_message: str, user_id: str):
    """
    Handles adding items to the cart contextually, resolving vague mentions and affirmations using fuzzy matching.
    """
    import re
    from services.llm_response import get_products
    from intents.cart_logic import get_cart_items
    
    session_products = get_products(user_id)
    current_cart     = get_cart_items(user_id)
    
    # --- Python-Level Context Resolution (Fuzzy Match / Affirmations) ---
    msg_lower = user_message.lower().strip()
    resolved_products = []
    
    # 1. Affirmations (Yes, add it, etc.) -> default to top match
    affirmations = ["yes", "yes please", "yep", "yeah", "add it", "add that", "sure", "ok", "okay", "that one", "the first one"]
    is_affirmation = any(msg_lower == aff for aff in affirmations) or msg_lower.startswith("yes")
    
    if session_products:
        if is_affirmation:
            # Add top result
            resolved_products.append(session_products[0])
        else:
            # 2. Fuzzy Matching
            user_tokens = set(re.findall(r'\w+', msg_lower)) - {"add", "the", "to", "cart", "please", "my", "some", "a", "an", "i", "want"}
            
            scored_products = []
            for p in session_products:
                p_name_lower = p['product_name'].lower()
                p_tokens = set(re.findall(r'\w+', p_name_lower))
                
                # Check for substring match
                substring_match = any(user_token in p_name_lower for user_token in user_tokens if len(user_token) > 2)
                
                # Token overlap
                overlap = len(user_tokens.intersection(p_tokens))
                
                score = overlap * 2
                if substring_match:
                    score += 1
                if p_name_lower in msg_lower:
                    score += 5
                    
                if score > 0:
                    scored_products.append((score, p))
            
            if scored_products:
                # Sort by score descending
                scored_products.sort(key=lambda x: x[0], reverse=True)
                top_score = scored_products[0][0]
                # Filter to top matches (in case of ties)
                resolved_products = [p for s, p in scored_products if s == top_score]
                
                # If they said "all of them", add everything
                word_tokens = re.findall(r'\w+', msg_lower)
                if "all" in word_tokens or "both" in word_tokens:
                    resolved_products = session_products

    product_context = ""
    if session_products:
        product_context += "RECENT SEARCH RESULTS:\n"
        for p in session_products:
            product_context += f"- {p['product_name']} (${p['price']})\n"
            
    if resolved_products:
        product_context += f"\nSYSTEM HAS RESOLVED USER'S REQUEST TO EXACTLY THESE PRODUCT(S):\n"
        for p in resolved_products[:1]:  # Ensure we just grab the best match to avoid confusion
            product_context += f"-> {p['product_name']} (${p['price']})\n"
        product_context += "\nCRITICAL: DO NOT ASK FOR CLARIFICATION! The system has successfully resolved the vague item to the correct match above. Output valid JSON adding it immediately.\n"

    cart_context = ""
    if current_cart:
        cart_context = "CURRENT CART CONTENT:\n"
        for i in current_cart:
            cart_context += f"- {i['product_name']} (x{i['quantity']})\n"
    
    system_prompt = f"""
    You are a premium grocery store assistant. Your tone is helpful, polite, and efficient.
    The user wants to add something to their cart.
    
    User Message: "{user_message}"
    
    {product_context}
    
    {cart_context}
    
    TASK:
    1. Check if the SYSTEM HAS RESOLVED the product. If yes, you MUST USE the resolved product(s) exactly. Do NOT ask for clarification. Proceed automatically.
    2. If no resolved products are provided, identify which product(s) the user wants from the "RECENT SEARCH RESULTS" based on their message.
    3. If they clearly said "all" or "both", add all from the recent search.
    4. If the product is entirely missing from context but is a specific request, you can add it if confident or ask for clarification.
    5. Always return JSON.
    6. Ensure the "message" is friendly and confirms what was added, without asking follow-up questions about the item just added.
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Perfect! I've added [Product] to your cart.",
        "added_products": [
            {{"product_name": "exact name", "quantity": 1, "price": 0.00}}
        ],
        "action_ready": true
    }}
    """
    return system_prompt


async def prepare_remove_from_cart_response(entities: Entity, user_message: str, user_id: str):
    """
    Handles removing items from the cart contextually.
    """
    from intents.cart_logic import get_cart_items
    current_cart = get_cart_items(user_id)
    
    cart_context = "CURRENT CART CONTENT:\n"
    if current_cart:
        for i in current_cart:
            cart_context += f"- {i['product_name']} (x{i['quantity']})\n"
    else:
        cart_context += "[Empty]"
    
    return f"""
    You are a premium grocery store assistant. The user wants to remove something from their cart.
    
    User Message: "{user_message}"
    
    {cart_context}
    
    TASK:
    1. Identify which exact product(s) the user wants to remove from the current cart.
    2. If multiple or "all", list them all.
    3. Provide a friendly confirmation message like "Consider it done! I've removed [Product] from your cart."
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Removal message",
        "removed_products": [
            {{"product_name": "exact name from context"}}
        ],
        "action_ready": true
    }}
    """


async def prepare_view_cart_response(user_id: str):
    """
    Returns the current state of the cart.
    """
    from intents.cart_logic import cart_summary
    summary = cart_summary(user_id)
    
    return """
    You are a helpful grocery store assistant.
    The user wants to see their cart.
    
    Current Cart Summary:
    {summary}
    
    TASK:
    1. Present the cart summary in a friendly way.
    2. If the cart is empty, suggest some popular items or ask what they need.
    3. If the cart has items, mention the total cost and ask if they are ready to checkout or if they need anything else.
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Detailed cart summary here",
        "action_ready": false
    }}
    """.format(summary=summary)


# ─────────────────────────────────────────────
#  RECIPE RESPONSES
# ─────────────────────────────────────────────
 
async def prepare_check_recipe_availability_response(entities: Entity, user_message: str) -> str:
    """
    Checks if ingredients for a recipe are in stock and builds a prompt
    for the LLM to respond to the user.
    """
    from intents.check_recipe_availability import check_recipe_availability
 
    result = check_recipe_availability(entities)
 
    if result.get("error"):
        context = result["error"]
    else:
        available   = [i["name"] for i in result["available"]]
        unavailable = [i["name"] for i in result["unavailable"]]
        context = (
            f"Recipe: {result['recipe_name']}\n"
            f"In stock:    {', '.join(available) if available else 'None'}\n"
            f"Out of stock: {', '.join(unavailable) if unavailable else 'None'}\n"
            f"Fully available: {result['fully_available']}"
        )
 
    return """
    You are a helpful grocery store assistant. Always respond in English.
    The user's message is: "{user_message}"
 
    Here is the recipe availability report:
    {context}
 
    YOUR TASK:
    1. Tell the user clearly whether they can make this recipe with what we have.
    2. List what ingredients are in stock and what's missing.
    3. If some ingredients are missing, suggest they may still be able to make it or find substitutes.
    4. Be friendly and conversational.
 
    OUTPUT FORMAT (JSON only, no other text):
    {{
        "message": "A friendly message summarising availability",
        "available": ["ingredient1", "ingredient2"],
        "unavailable": ["ingredient3"],
        "fully_available": true/false,
        "action_ready": false
    }}
    """.format(user_message=user_message, context=context)
 


async def prepare_budget_recipe_response(entities, user_message: str) -> str:
    """
    Ask context questions before recommending recipes.
    Understand: cooking style, cuisine preference, meal type, dietary restrictions, etc.
    """
    from intents.budget_recipe_suggestion import budget_recipe_suggestion
    
    budget = entities.budget
    
    # Check if user provided any context clues
    cuisine_keywords = ["italian", "jamaican", "asian", "mexican", "indian", "caribbean", "american", "chinese", "spanish"]
    meal_keywords = ["breakfast", "lunch", "dinner", "snack", "appetizer", "dessert", "main", "side"]
    diet_keywords = ["vegan", "vegetarian", "gluten", "healthy", "quick", "easy", "simple", "light"]
    
    has_cuisine = any(keyword in user_message.lower() for keyword in cuisine_keywords)
    has_meal = any(keyword in user_message.lower() for keyword in meal_keywords)
    has_diet = any(keyword in user_message.lower() for keyword in diet_keywords)
    
    # If no context provided, ask first
    if not (has_cuisine or has_meal or has_diet):
        system_prompt = f"""
        You are a helpful, friendly grocery store assistant.
        User message: "{user_message}"
        User budget: ${budget}
        
        CONTEXT: The user has a budget and wants recipe ideas, but hasn't specified their preferences yet.
        
        YOUR TASK: Ask clarifying questions to understand what they want to cook!
        
        Ask about:
        1. What meal (breakfast, lunch, dinner, snack, or something special)
        2. Any cuisine they prefer (Jamaican, Italian, quick & easy, comfort food, healthy, etc.)
        3. Who they're cooking for (just themselves, family, guests)
        4. Any dietary preferences or restrictions
        
        BE CONVERSATIONAL: Don't ask as a list. Make it sound natural and friendly!
        
        Example: "Perfect! With $2000 you can make something really tasty! Before I suggest recipes, tell me - what kind of meal are you thinking? Like a quick weeknight dinner, something fancy for guests, or maybe breakfast? And do you have any preferences - are you in the mood for something light and healthy, or more comfort food?"
        
        OUTPUT (JSON only):
        {{
            "message": "Your conversational question about their preferences",
            "action_ready": false
        }}
        """
        return system_prompt
    
    # If we have context, get recommendations
    result = budget_recipe_suggestion(entities)
    
    if result.get("error") or not result["suggestions"]:
        context = f"No recipes found within ${budget} budget."
    else:
        lines = []
        for r in result["suggestions"][:5]:
            missing = ""
            if r.get("missing_ingredients"):
                missing_list = r["missing_ingredients"][:2]
                missing = f" (just need: {', '.join(missing_list)})"
            lines.append(f"• {r['recipe_name']} — ${r['estimated_cost']} | Serves {r['servings']}{missing}")
        context = "\n".join(lines)
    
    system_prompt = f"""
    You are a friendly, enthusiastic grocery store assistant.
    User budget: ${budget}
    User said: "{user_message}"
    
    Recipe suggestions within their budget:
    {context}
    
    INSTRUCTIONS:
    - Be warm and conversational (not robotic)
    - Talk ABOUT the recipes, don't just list them
    - Pick 2-3 favorites and describe why they're great
    - Mention they're within budget
    - If they're missing ingredients, frame positively: "You'll just need to pick up X"
    - Ask which one sounds good or if they want more options
    - NO BULLET POINTS - write naturally!
    
    OUTPUT (JSON only):
    {{
        "message": "Conversational response. Example: 'Great choices within your budget! The Shepherd's Pie is perfect if you're feeding a family - hearty, delicious, and only costs $270. If you want something quicker, the Fried Plantain is amazing and super affordable at $320. Both are crowd-pleasers! Which sounds better to you, or should I suggest something else?'",
        "suggestions": [
            {{"recipe_name": "name", "estimated_cost": 0.00, "servings": 4}}
        ],
        "action_ready": false
    }}
    """
    
    return system_prompt 

 
async def prepare_recommend_recipe_response(entities: Entity, user_message: str, recommendations: dict = None) -> str:
    """
    Recommends recipes based on ingredients the user mentioned or items in their cart.
    Now receives actual recipe data from recommend_recipe().
    """
    
    # Use the recommendations data if provided
    if not recommendations:
        from intents.recommend_recipe import recommend_recipe
        recommendations = recommend_recipe(entities)
    
    # Build context from actual results
    if recommendations.get("error"):
        context = recommendations["error"]
        based_on = "your request"
    elif not recommendations.get("recommendations") or len(recommendations.get("recommendations", [])) == 0:
        context = f"No recipes found using: {recommendations.get('based_on', 'your ingredients')}."
        based_on = recommendations.get('based_on', 'your ingredients')
    else:
        lines = []
        recs = recommendations.get("recommendations", [])[:5]
        
        for i, r in enumerate(recs, 1):
            missing = ""
            if r.get('unavailable_ingredients'):
                missing_list = r['unavailable_ingredients'][:2]
                missing = f" (missing: {', '.join(missing_list)})"
            
            lines.append(f"{i}. {r['recipe_name']} - {r['match_score']}% match{missing}")
        
        context = f"Based on: {recommendations.get('based_on')}\n\nRecipe suggestions:\n" + "\n".join(lines)
        based_on = recommendations.get('based_on', 'your ingredients')

    return f"""
    You are a friendly grocery store assistant recommending recipes.
    The user has these items in their cart: {based_on}
    The user's message is: "{user_message}"

    Here are recipe recommendations based on their cart:
    {context}
    
    INSTRUCTIONS:
    - Output ONLY valid JSON
    - The "message" field must be conversational and natural (no lists, no bullets, no asterisks)
    - Mention the recipes by name
    - Be warm and enthusiastic
    - If recipes were found, suggest one or two favorites
    - If no recipes were found, create a simple recipe using the items in their cart plus a few common pantry ingredients (like salt, pepper, oil, or onion)
    - Do not keep asking for more ingredients - just work with what you have and be creative!
    - You can always suggest a recipe even if some ingredients are missing - just be honest about what's not available and keep it positive. For example, "I found a great recipe for Chicken Stir Fry! It usually calls for bell peppers, but you could still make it without them and it would be delicious!"
    - Ask if they want the full recipe for any
    - DONT MENTION MATCH SCORE IN THE MESSAGE - it's just for internal ranking, not for the user to see

    OUTPUT FORMAT (JSON only, no other text):
    {{
        "message": "Natural conversational response. For example: 'Great! With chicken and rice in your cart, I found some delicious options! You could make a tasty Jamaican Jerk Chicken with Rice or try Chicken Biryani. Both are amazing! Which would you like the full recipe for?'",
        "recommendations": [
            {{
                "recipe_name": "name",
                "match_score": 85.0,
                "servings": 4
            }}
        ],
        "action_ready": false
    }}
    """
 

async def prepare_get_recipe_response(entities: Entity, user_message: str) -> str:
    """
    Fetches a specific recipe and builds a prompt for the LLM to
    present it in a friendly way.
    """
    from intents.get_recipe import get_recipe_data
 
    recipe_data = get_recipe_data(entities)
 
    if not recipe_data:
        context = f"No recipe found for: {entities.recipe_name}"
    else:
        r = recipe_data[0]
        context = (
            f"Recipe: {r['recipe_name']}\n"
            f"Serves: {r['servings']}\n"
            f"Ingredients: {r['ingredients_with_amounts']}\n"
            f"Instructions: {r['instructions']}"
        )
 
    return """
You are a helpful grocery store assistant. Always respond in English.
The user's message is: "{user_message}"

Here is the recipe data:
{context}

YOUR TASK:
1. Present the recipe in a clear, friendly way.
2. List ingredients with their amounts.
3. Break the instructions into readable steps.
4. Ask if they'd like to add any ingredients to their cart.
5. If no recipe data exists or the recipe cannot be fully matched, create a simple recipe using items from the user's cart plus common pantry staples (like salt, pepper, oil, onion, garlic). Do not ask the user to provide additional ingredients.
6. Keep the response natural and friendly.

OUTPUT FORMAT (JSON only, no other text):
{{
    "message": "A friendly intro to the recipe, e.g., 'Here’s a simple Chicken and Rice Bowl I created using the ingredients in your cart along with some pantry staples. Would you like me to add these to your cart?'",
    "recipe_name": "name",
    "servings": 4,
    "ingredients": [{{"name": "...", "amount": "..."}}],
    "steps": ["Step 1: ...", "Step 2: ..."],
    "action_ready": false
}}
""".format(user_message=user_message, context=context)


async def prepare_aisle_location_response(entities, user_message: str) -> str:
    """
    Prepare system prompt for aisle/location queries.
    User asks: "Where can I find bread?" or "What aisle is milk in?"
    """
    from intents.stock_check import get_product_aisle
    
    product_name = entities.product_name
    aisle_info = get_product_aisle(product_name) if product_name else None
    
    if not aisle_info:
        context = f"Product '{product_name}' not found in our inventory."
        aisle_number = None
    else:
        context = f"""
Product: {aisle_info['product_name']}
Category: {aisle_info['category']}
Aisle: {aisle_info['aisle']}
Location Details: {aisle_info['location_details']}
"""
        aisle_number = aisle_info.get('aisle')
    
    system_prompt = f"""
    You are a friendly and helpful UCC Supermarket store assistant.
    A customer is asking where to find a product in the store.
    
    Customer question: "{user_message}"
    
    Product location information:
    {context}
    
    INSTRUCTIONS:
    - Be warm and friendly
    - Give clear, easy-to-follow directions
    - If product not found, apologize and suggest they ask a staff member
    - Always mention aisle number if available
    - Include location details (left side, back wall, etc)
    - Keep response short and natural
    - Do NOT use lists or bullet points
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Your friendly directions here. For example: 'You'll find our bread in Aisle 5 on the right side of the store, usually on the middle shelf. Let me know if you need help finding anything else!'",
        "aisle": {aisle_number},
        "action_ready": false
    }}
    """
    
    return system_prompt


# Prepare store info response



async def prepare_add_inventory_to_cart_response(product_names: List[str], user_message: str) -> str:
    """
    Prepare response for adding inventory items to cart.
    """
    import json
    if not product_names:
        return json.dumps({"message": "No products selected. Which product would you like to add?"})
    
    product_str = ", ".join(product_names) if len(product_names) <= 3 else f"{', '.join(product_names[:3])}, and more"
    
    return f"""
    You are a helpful UCC Supermarket assistant.
    User wants to add these items to cart: {product_str}
    User message: "{user_message}"
    
    INSTRUCTIONS:
    - Confirm the items they want to add
    - Ask for quantities if not specified
    - Be helpful and quick
    - Format the response naturally
    
    OUTPUT (JSON only):
    {{
        "message": "Perfect! I'll add {product_names[0]} to your cart. How many would you like?",
        "products_to_add": {product_names},
        "action_ready": true
    }}
    """
    return inventory_prompt


async def prepare_inventory_selection_response(user_message: str, options: list):
    """
    LLM prompt to extract the exact selection from a list of options.
    Handles typos, ordinals, and descriptions.
    """
    options_str = "\n".join([f"- {opt}" for opt in options])
    
    prompt = f"""
    You are an expert intent extraction engine for a supermarket.
    The user is trying to select an item from a list of previously suggested options.
    
    USER MESSAGE: "{user_message}"
    
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
        "selected_inventory": "exact_name_from_options_or_None",
    }}
    """
    return prompt