"""
recommend_recipe.py
Now correctly calculates match_score based on what user HAS IN CART
"""
from venv import logger

from datasets.data import recipes_df, carts
from models.llm_classes import Entity
from intents.check_recipe_availability import extract_ingredients
from intents.stock_check import get_inventory_data

def extract_main_keywords(text: str) -> list:
    """
    Extract main ingredient keywords, ignoring descriptors.
    
    Examples:
    - "Boneless Skinless Chicken Breast" → ["chicken"]
    - "Jasmine Rice" → ["rice"]
    - "Ground Beef" → ["beef"]
    - "Chicken and Rice" → ["chicken", "rice"]
    """
    descriptors = {
        'boneless', 'skinless', 'jasmine', 'ground', 'fresh', 'frozen',
        'cooked', 'raw', 'organic', 'whole', 'sliced', 'diced', 'minced',
        'medium', 'large', 'small', 'sweet', 'white', 'brown', 'red',
        'dried', 'canned', 'breaded', 'seasoned', 'unseasoned'
    }
    
    parts = text.replace(" and ", ",").replace(" or ", ",").split(",")
    
    keywords = []
    for part in parts:
        words = part.strip().lower().split()
        meaningful_words = [w for w in words if w not in descriptors]
        
        if meaningful_words:
            keywords.append(meaningful_words[-1])
    
    return keywords


def get_user_cart_products(user_id: str) -> list:
    """Get product names from user's cart"""
    user_cart = carts.get(user_id, [])
    return [item['product_name'].lower() for item in user_cart]


def normalize_ingredient_name(ingredient_name: str) -> str:
    """Normalize ingredient name for comparison"""
    return ingredient_name.lower().strip()


def ingredient_in_user_cart(ingredient_name: str, user_cart_products: list) -> bool:
    """
    Check if an ingredient is in the user's cart.
    Handles fuzzy matching for variations like "Shrimp" vs "Shrimp - Fresh"
    """
    normalized_ingredient = normalize_ingredient_name(ingredient_name)
    
    for cart_product in user_cart_products:
        # Extract main ingredient name (before dash if present)
        cart_main = cart_product.split(' - ')[0]
        
        # Direct match
        if normalized_ingredient in cart_main.lower():
            return True
        if cart_main.lower() in normalized_ingredient:
            return True
    
    return False


def recommend_recipe(entities: Entity, user_id: str = None, user_message: str = "") -> dict:
    """
    Recommends recipes based on what the user HAS in their cart.
    Calculates match_score as: (ingredients user HAS) / (total ingredients needed) * 100
    
    Returns:
        {
            "based_on": str,
            "recommendations": [
                {
                    "recipe_name": str,
                    "category": str,
                    "servings": int,
                    "available_ingredients":   [str],      # What user HAS
                    "unavailable_ingredients": [str],      # What user NEEDS to buy
                    "match_score": float,
                    "instructions": str
                }
            ],
            "error": str | None
        }
    """
    ingredient_hint = entities.product_name or entities.recipe_name
    
    #   If user doesn't mention ingredients, use items from their cart
    if not ingredient_hint and user_id:
        user_cart = carts.get(user_id, [])
        if user_cart:
            cart_products = [item['product_name'] for item in user_cart]
            ingredient_hint = " and ".join(cart_products)
            logger.info(f"[RECOMMEND] Using cart items: {ingredient_hint}")
        else:
            return {
                "error": "Please mention an ingredient or add items to your cart.",
                "based_on": None,
                "recommendations": []
            }
    
    if not ingredient_hint:
        return {
            "error": "Please mention an ingredient to get recommendations.",
            "based_on": None,
            "recommendations": []
        }
    
    #   Extract keywords and get user's cart products
    keywords = extract_main_keywords(ingredient_hint)
    keywords = [k for k in keywords if k]
    
    user_cart_products = get_user_cart_products(user_id) if user_id else []
    
    matched_recipes = []
    
    for _, row in recipes_df.iterrows():
        ingredients_raw = row["ingredients_with_amounts"].lower()
        
        #   Recipe must contain AT LEAST ONE keyword
        if not any(kw in ingredients_raw for kw in keywords):
            continue
        
        ingredients = extract_ingredients(row["ingredients_with_amounts"])
        
        #   CRITICAL: Calculate match based on what USER HAS IN CART
        available_in_cart = []      # Ingredients user already has
        unavailable_needed = []     # Ingredients user needs to buy
        
        for ing in ingredients:
            ingredient_name = ing["name"]
            
            # Check if user has this ingredient in their cart
            if ingredient_in_user_cart(ingredient_name, user_cart_products):
                available_in_cart.append(ingredient_name)
            else:
                unavailable_needed.append(ingredient_name)
        
        #   Match score = (ingredients user HAS) / (total ingredients) * 100
        total = len(ingredients)
        match_score = round(len(available_in_cart) / total * 100, 1) if total > 0 else 0
        
        matched_recipes.append({
            "recipe_name": row["recipe_name"],
            "category": row.get("category", ""),
            "servings": row.get("servings", ""),
            "available_ingredients": available_in_cart,
            "unavailable_ingredients": unavailable_needed,
            "match_score": match_score,
            "instructions": row["instructions"]
        })
    
    # Sort by match score
    matched_recipes.sort(key=lambda x: x["match_score"], reverse=True)
    
    for recipe in matched_recipes[:3]:
        logger.info(f"  - {recipe['recipe_name']}: {recipe['match_score']}% "
                   f"(has: {len(recipe['available_ingredients'])}, needs: {len(recipe['unavailable_ingredients'])})")
    
    return {
        "based_on": ingredient_hint,
        "recommendations": matched_recipes[:5],
        "error": None
    }