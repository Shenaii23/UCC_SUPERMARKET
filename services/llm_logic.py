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
    5. Provide a friendly, natural response that mentions the products you found, their prices, and quantities. For example, "We have whole milk ($350) and chocolate milk ($400) in stock!"
    6. If no products are found, respond with a friendly message saying you couldn't find any matches, and maybe suggest checking the spelling or trying a different search term.
    7. If you have a variety of matches, you can group them in your response. For example, "We have several milk options available: whole milk ($350), skim milk ($300), chocolate milk ($400), and cashew milk ($450)!"
    8. In your response, ask the user if they would like to add any of the found products to their cart.

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
        "message": "We have Apple Sauce ($150) and Unsweetened Applesauce ($120) in stock. Would you like to add either of these to your cart?",
        "products": [
            {{"product_name": "Apple Sauce", "quantity": 25, "price": 150}},
            {{"product_name": "Unsweetened Applesauce", "quantity": 10, "price": 120}}
        ],
        "action_ready": true
    }}

    User searched: "ackee"
    Inventory: [Ackee - Canned, Ackee - Frozen, Blackberries - Fresh]
    {{
        "message": "We have ackee available! We've got canned ($450) and frozen ($550) options. Would you like to add either of these to your cart?",
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
    Handles adding items to the cart contextually.
    """
    from services.llm_response import get_products
    from intents.cart_logic import get_cart_items
    
    # Check if we have products in the current session (from a recent stock_check)
    session_products = get_products(user_id)
    current_cart     = get_cart_items(user_id)
    
    product_context = ""
    if session_products:
        product_context = "RECENT SEARCH RESULTS (User might be referring to these):\n"
        for p in session_products:
            product_context += f"- {p['product_name']} (${p['price']})\n"
    
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
    1. Identify which exact product(s) the user wants to add based on the message and context.
    2. If the user provided a name that matches or nearly matches a product in the "RECENT SEARCH RESULTS", use that exact product name and price.
    3. If they said "all" or mentioned multiple, add all of them.
    4. If the product is not in the context, BUT it's a very specific request, you can still add it if you're confident, or ask for clarification.
    5. Always return JSON.
    6. Ensure the "message" is friendly and confirms what was added. Mention the total items in the cart now if appropriate.

    IMPORTANT:
    - if the users last intent was recipe recommendation and the user asks for the item 
    to be placed in the cart (e.g. "add the first one to my cart"), make sure to check the products in the recent search results
    first return a message that will ask for confirmation of the products before adding to the cart. For example,
      "Just to confirm, you want to add [Product Name] to your cart, is that correct?" 
    
    OUTPUT FORMAT (JSON only):
    {{
        "message": "Perfect! I've added [Product] to your cart. You now have [N] items in your basket.",
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
 
 
async def prepare_budget_recipe_response(entities: Entity, user_message: str) -> str:
    """
    Finds recipes within the user's budget and builds a prompt
    for the LLM to respond with suggestions.
    """
    from intents.budget_recipe_suggestion import budget_recipe_suggestion
 
    result = budget_recipe_suggestion(entities)
 
    if result.get("error"):
        context = result["error"]
    elif not result["suggestions"]:
        context = f"No recipes found within budget of ${result['budget']}."
    else:
        lines = []
        for r in result["suggestions"]:
            missing = f" (missing: {', '.join(r['missing_ingredients'])})" if r["missing_ingredients"] else ""
            lines.append(
                f"- {r['recipe_name']} | Est. cost: ${r['estimated_cost']} | Serves: {r['servings']}{missing}"
            )
        context = f"Budget: ${result['budget']}\n" + "\n".join(lines)
 
    return """
    You are a grocery store assistant API.
    The user's message is: "{user_message}"
 
    Here are recipe suggestions within the user's budget:
    {context}
 
    YOUR TASK:
    1. Present the recipe options in a friendly, enthusiastic way.
    2. Mention the estimated cost and servings for each.
    3. If any ingredients are missing, mention it briefly but keep it positive.
    4. If no recipes were found, apologise and suggest a higher budget.
 
    OUTPUT FORMAT (JSON only, no other text):
    {{
        "message": "A friendly message listing the recipe suggestions",
        "suggestions": [
            {{
                "recipe_name": "name",
                "estimated_cost": 0.00,
                "servings": 4
            }}
        ],
        "action_ready": false
    }}
    """.format(user_message=user_message, context=context)
 
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
