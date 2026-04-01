# services/llm_logic.py
from models.llm_classes import Entity

# intent files
from intents.check_recipe_availability import get_recipe_data
from intents.stock_check import get_inventory_data
from services.context_handler import update_user_state, get_user_state

#from intents.cart_logic import add_to_cart, remove_from_cart, cart_summary, cart_statement, clear_cart
#from intents.get_recipe import get_recipe_logic
#from intents.budget_recipe_suggestion import budget_recipe_suggestion_logic
#from intents.recommend_recipe import recommend_recipe_logic
#from intents.general_chat import general_chat_logic

# Stock checking function that will be used in the llm_route to check the inventory 
# for the given product name and return the available quantity, prices etc.
async def prepare_stock_response(entities: Entity, user_message: str):
    """
    Docstring for prepare_stock_response
    
    :param entities: The entities extracted from the user's message
    :type entities: Entity
    :param user_message: The original user message
    :type user_message: str
    This function checks the inventory for the given product name
      and returns the available quantity.
    """


    # Use the cleaned data from the pydantic model to search the inventory.
    product_to_find = entities.product_name

    # Get the category hint
    category = entities.category_hint

    # Get the data from the get inventory_data function to use in the system prompt for the LLM.
    results = get_inventory_data(product_to_find, category)

    # Format the results into sting for the llm 
    if not results:
        context = f"No matching products found in inventory for search term: {product_to_find}"
    else:
        context = "Found these items:\n" + "\n".join([str(r) for r in results])

    # Set up the system prompt for the LLM with the context and instructions for intent detection.
    inventory_prompt = """
    You are a helpful grocery store assistant. 

    The users original message is: "{user_message}"

    The user searched for: "{product_to_find}"

    Here are the matching products from our inventory:
    {context}

    YOUR TASK:
    1. Select products that match what exactly what the user asked for.
    2. When in doubt, INCLUDE the product - let the user decide
    3. Only exclude products that are clearly unrelated (e.g., "apple juice" when searching for "apple sauce")
    4. Include all variants and related products (e.g., whole milk, skim milk, chocolate milk, cashew milk all count as "milk")
    5. Provide a friendly, natural response. ALWAYS list the products you found, their prices, and quantities using a Markdown bulleted list, with each product on its own line.
    6. If no products are found, respond with a friendly message saying you couldn't find any matches, and maybe suggest checking the spelling or trying a different search term.
    7. If you have a variety of matches, introduce them and then list them out using Markdown bullets. For example:
       "We have several milk options available! Here's what we found:
       - Whole milk ($350)
       - Skim milk ($300)
       - Chocolate milk ($400)
       - Cashew milk ($450)"
    8. At the end of your response, ask the user if they would like to add any of the found products to their cart.

    OUTPUT FORMAT (JSON only, no other text):
    {{
        "message": "A friendly message to show the user",
        "products": [
            {{
                "product_name": "exact name from inventory",
                "quantity": available quantity,
                "price": price
            }}
        ],
        "action_ready": true/false  // true if these products can be added to cart directly
    }}

    EXAMPLES:
    
    User searched: "apple sauce"
    Inventory: [Apple Sauce, Unsweetened Applesauce, Apple Juice]
    {{
        "message": "We have a few apple sauce options in stock:\\n- Apple Sauce ($150)\\n- Unsweetened Applesauce ($120)\\n\\nWould you like to add either of these to your cart?",
        "products": [
            {{"product_name": "Apple Sauce", "quantity": 25, "price": 150}},
            {{"product_name": "Unsweetened Applesauce", "quantity": 10, "price": 120}}
        ],
        "action_ready": true
    }}
    
    User searched: "ackee"
    Inventory: [Ackee - Canned, Ackee - Frozen, Blackberries - Fresh]
    {{
        "message": "We have ackee available! We've got:\\n- Canned Ackee ($450)\\n- Frozen Ackee ($550)\\n\\nWould you like to add either of these to your cart?",
        "products": [
            {{"product_name": "Ackee - Canned", "quantity": 35, "price": 450}},
            {{"product_name": "Ackee - Frozen", "quantity": 20, "price": 550}}
        ],
        "action_ready": true
    }}
    
    Now process the user's search.
    """.format(user_message=user_message, product_to_find=product_to_find, context=context)
    
    return inventory_prompt

# Add to cart logic that will be used in the llm_route to add items to the cart based on the user's message 
# and the detected intent.

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

async def prepare_store_info_response(entities, user_message: str) -> str:
    """
    Prepare system prompt for store info queries.
    User asks: "What are your store hours?" or "Where is the pharmacy?"
    """
    from intents.store_info import get_store_info
    
    store_info = get_store_info()
    
    context = f"""
Store Information:
- Name: {store_info['name']}
- Address: {store_info['address']}
- Phone: {store_info['phone']}
- Hours: {store_info['hours']}
- Departments: {', '.join(store_info['departments'])}
"""
    
    system_prompt = f"""
    You are a friendly and helpful UCC Supermarket store assistant.
    A customer is asking for information about the store.
    
    Customer question: "{user_message}"
    
    Store information:
    {context}
    
    INSTRUCTIONS:
    - Be warm and friendly
    - Provide clear, concise answers
    - Mention store name, address, phone, hours, and departments
    - Keep response short and natural
    - Do NOT use lists or bullet points
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Your friendly response here. For example: 'Welcome to UCC Supermarket! We're located at 123 Main Street and are open Monday-Saturday from 8am-9pm.
         You can reach us at 555-1234. We have departments like produce, dairy, meat, and a pharmacy.
         Let me know if you need anything else!'",
        "action_ready": false
    }}
    """
    
    return system_prompt




async def prepare_add_inventory_to_cart_response(product_names: List[str], user_message: str) -> str:
    """
    Prepare response for adding inventory items to cart.
    """
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