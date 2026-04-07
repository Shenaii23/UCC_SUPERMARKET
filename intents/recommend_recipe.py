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


from typing import List, Dict, Optional
from rapidfuzz import fuzz
import logging

logger = logging.getLogger(__name__)

def recommend_recipe(entities: Entity, user_id: str = None, user_message: str = "") -> dict:
    """
    Recommends recipes based on what the user HAS in their cart.
    Calculates match_score as: (ingredients user HAS) / (total ingredients needed) * 100
    
    IMPROVEMENTS:
    - Better keyword extraction from cart items
    - Fuzzy matching for ingredients (handles "Chicken Breast" vs "boneless skinless chicken breast")
    - Ranked by match score and ingredient variety
    - Returns more recommendations (up to 10)
    
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
    
    # If user doesn't mention ingredients, use items from their cart
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
    
    # ========== IMPROVED: Extract keywords from cart items ==========
    keywords = extract_cart_keywords(ingredient_hint)
    logger.info(f"[RECOMMEND] Extracted keywords: {keywords}")
    
    user_cart_products = get_user_cart_products(user_id) if user_id else []
    logger.info(f"[RECOMMEND] User cart products: {user_cart_products}")
    
    matched_recipes = []
    
    for _, row in recipes_df.iterrows():
        ingredients_raw = row["ingredients_with_amounts"].lower()
        recipe_name = row["recipe_name"]
        
        # ========== IMPROVED: Better keyword matching ==========
        # Check if recipe contains AT LEAST ONE keyword (more lenient)
        keyword_match = False
        for kw in keywords:
            if kw.lower() in ingredients_raw:
                keyword_match = True
                logger.debug(f"  ✓ Recipe '{recipe_name}' matches keyword '{kw}'")
                break
        
        if not keyword_match:
            logger.debug(f"  ✗ Recipe '{recipe_name}' skipped - no keyword match")
            continue
        
        ingredients = extract_ingredients(row["ingredients_with_amounts"])
        
        # ========== IMPROVED: Fuzzy matching for ingredients ==========
        available_in_cart = []      # Ingredients user already has
        unavailable_needed = []     # Ingredients user needs to buy
        
        for ing in ingredients:
            ingredient_name = ing["name"]
            
            # Use fuzzy matching instead of exact matching
            has_ingredient = ingredient_in_user_cart_fuzzy(
                ingredient_name, 
                user_cart_products,
                threshold=70  # 70% match is good enough
            )
            
            if has_ingredient:
                available_in_cart.append(ingredient_name)
                logger.debug(f"    ✓ User HAS: {ingredient_name}")
            else:
                unavailable_needed.append(ingredient_name)
                logger.debug(f"    ✗ User NEEDS: {ingredient_name}")
        
        # Match score = (ingredients user HAS) / (total ingredients) * 100
        total = len(ingredients)
        match_score = round(len(available_in_cart) / total * 100, 1) if total > 0 else 0
        
        matched_recipes.append({
            "recipe_name": recipe_name,
            "category": row.get("category", ""),
            "servings": row.get("servings", ""),
            "available_ingredients": available_in_cart,
            "unavailable_ingredients": unavailable_needed,
            "match_score": match_score,
            "instructions": row["instructions"],
            "ingredient_count": len(ingredients)  # For secondary sort
        })
        
        logger.info(f"  ✓ '{recipe_name}': {match_score}% match "
                   f"(has: {len(available_in_cart)}/{total})")
    
    # ========== IMPROVED: Smarter sorting ==========
    # Sort by:
    # 1. Match score (higher is better)
    # 2. Number of ingredients user HAS (higher is better - more concrete matches)
    # 3. Fewer ingredients needed (lower is better - easier to complete)
    matched_recipes.sort(
        key=lambda x: (
            x["match_score"],                              # Primary: match %
            len(x["available_ingredients"]),               # Secondary: ingredients user HAS
            -len(x["unavailable_ingredients"])             # Tertiary: fewer needed
        ),
        reverse=True
    )
    
    logger.info(f"[RECOMMEND] Found {len(matched_recipes)} matching recipes")
    for i, recipe in enumerate(matched_recipes[:5], 1):
        logger.info(f"  {i}. {recipe['recipe_name']}: {recipe['match_score']}% "
                   f"(has: {len(recipe['available_ingredients'])}, needs: {len(recipe['unavailable_ingredients'])})")
    
    # Return up to 10 recommendations instead of 5
    return {
        "based_on": ingredient_hint,
        "recommendations": matched_recipes[:10],  # Increased from 5 to 10
        "error": None
    }


# ========== HELPER: Improved keyword extraction from cart ==========
def extract_cart_keywords(cart_items_str: str) -> List[str]:
    """
    Extract meaningful keywords from cart items string.
    
    Examples:
    - "Boneless Skinless Chicken Breast" → ["chicken", "breast", "meat"]
    - "Bread - Banana" → ["banana", "bread"]
    - "Cheese - Cream Cheese Block" → ["cheese", "cream cheese"]
    - "Curry Leaves - Fresh" → ["curry", "leaves", "spice"]
    
    Args:
        cart_items_str: Space-separated or dash-separated product names
    
    Returns:
        List of keywords suitable for recipe matching
    """
    
    # Common category keywords (broad matches)
    category_keywords = {
        "chicken": ["chicken", "poultry", "meat"],
        "beef": ["beef", "steak", "meat"],
        "pork": ["pork", "ham", "meat"],
        "fish": ["fish", "seafood"],
        "shrimp": ["shrimp", "seafood"],
        "cheese": ["cheese", "dairy"],
        "milk": ["milk", "dairy"],
        "bread": ["bread", "flour", "baking"],
        "flour": ["flour", "baking"],
        "rice": ["rice", "grain"],
        "pasta": ["pasta", "noodle"],
        "tomato": ["tomato", "sauce"],
        "onion": ["onion", "vegetable"],
        "garlic": ["garlic", "spice"],
        "salt": ["salt", "seasoning", "spice"],
        "pepper": ["pepper", "spice"],
        "curry": ["curry", "spice", "indian"],
        "bbq": ["bbq", "sauce", "seasoning"],
    }
    
    keywords = set()
    
    # Split by common delimiters
    items = cart_items_str.replace(" - ", " ").replace(" and ", " ").split()
    
    for item in items:
        item_lower = item.lower().strip('.,')
        
        # Skip very short or stop words
        if len(item_lower) <= 1 or item_lower in {"a", "an", "the", "x", "with", "in"}:
            continue
        
        # Add the item itself
        keywords.add(item_lower)
        
        # Add related keywords
        for keyword, related in category_keywords.items():
            if keyword in item_lower:
                keywords.update(related)
    
    # Filter empty strings and return as list
    keywords = [kw for kw in keywords if kw.strip()]
    
    logger.debug(f"[EXTRACT_KEYWORDS] From '{cart_items_str}' got: {keywords}")
    return list(keywords)


# ========== HELPER: Improved ingredient matching with fuzzy logic ==========
def ingredient_in_user_cart_fuzzy(
    recipe_ingredient: str,
    user_cart_products: List[str],
    threshold: int = 70
) -> bool:
    """
    Check if user has a recipe ingredient in their cart using fuzzy matching.
    
    Examples:
    - "Chicken Breast" vs "Boneless Skinless Chicken Breast" → True (90% match)
    - "Cheese" vs "Cream Cheese Block" → True (fuzzy match)
    - "Milk" vs "Whole Milk" → True (substring match)
    - "Random Ingredient" vs cart items → False
    
    Args:
        recipe_ingredient: The ingredient name from the recipe
        user_cart_products: List of product names in user's cart
        threshold: Minimum match score (0-100) to consider it a match
    
    Returns:
        bool: True if user has this ingredient or something close to it
    """
    
    recipe_ing_lower = recipe_ingredient.lower().strip()
    
    for cart_product in user_cart_products:
        cart_prod_lower = cart_product.lower().strip()
        
        # ========== Exact substring match (best case) ==========
        if recipe_ing_lower in cart_prod_lower or cart_prod_lower in recipe_ing_lower:
            logger.debug(f"    [MATCH] Exact: '{recipe_ingredient}' ⊂ '{cart_product}'")
            return True
        
        # ========== Token-level match ==========
        recipe_tokens = set(recipe_ing_lower.split())
        cart_tokens = set(cart_prod_lower.split())
        
        # If all recipe tokens are in cart product (e.g., "chicken breast" in "boneless skinless chicken breast")
        if recipe_tokens.issubset(cart_tokens):
            logger.debug(f"    [MATCH] Tokens: {recipe_tokens} ⊂ {cart_tokens}")
            return True
        
        # ========== Fuzzy match (handles typos, variations) ==========
        fuzzy_score = fuzz.ratio(recipe_ing_lower, cart_prod_lower)
        partial_score = fuzz.partial_ratio(recipe_ing_lower, cart_prod_lower)
        
        # Use partial score for better matching (handles substrings)
        if partial_score >= threshold:
            logger.debug(f"    [MATCH] Fuzzy: '{recipe_ingredient}' vs '{cart_product}' = {partial_score}%")
            return True
        
        # ========== Semantic match (same base ingredient type) ==========
        # Group related ingredients
        ingredient_groups = {
            "chicken": ["chicken", "poultry", "fowl"],
            "beef": ["beef", "steak", "cow"],
            "pork": ["pork", "ham", "pig"],
            "fish": ["fish", "salmon", "tuna", "seafood"],
            "cheese": ["cheese", "cheddar", "mozzarella", "cream cheese"],
            "milk": ["milk", "dairy"],
            "bread": ["bread", "flour", "loaf"],
            "pasta": ["pasta", "noodle", "spaghetti"],
        }
        
        for group_name, group_members in ingredient_groups.items():
            recipe_has_group = any(member in recipe_ing_lower for member in group_members)
            cart_has_group = any(member in cart_prod_lower for member in group_members)
            
            if recipe_has_group and cart_has_group:
                logger.debug(f"    [MATCH] Semantic: Both are '{group_name}' type")
                return True
    
    logger.debug(f"    [NO MATCH] '{recipe_ingredient}' not in cart")
    return False


# ========== HELPER: Fallback (old exact matching, kept for compatibility) ==========
def ingredient_in_user_cart(ingredient_name: str, user_cart_products: List[str]) -> bool:
    """
    Original exact matching function (kept for backward compatibility).
    Use ingredient_in_user_cart_fuzzy() instead for better results.
    """
    ingredient_lower = ingredient_name.lower()
    for product in user_cart_products:
        if ingredient_lower in product.lower():
            return True
    return False


# ========== HELPER: Extract ingredients from recipe ==========
def extract_ingredients(ingredients_str: str) -> List[Dict[str, str]]:
    """
    Parse ingredients string and extract name and amount.
    
    Example:
    "1 cup flour, 2 eggs, salt to taste"
    →
    [
        {"name": "flour", "amount": "1 cup"},
        {"name": "eggs", "amount": "2"},
        {"name": "salt", "amount": "to taste"}
    ]
    """
    ingredients = []
    
    # Split by comma
    for item in ingredients_str.split(","):
        item = item.strip()
        if not item:
            continue
        
        # Try to separate amount from ingredient name
        parts = item.split(maxsplit=2)  # Split into max 3 parts
        
        if len(parts) >= 2:
            # Assume format: "amount unit ingredient"
            amount = parts[0]
            ingredient_name = " ".join(parts[2:]) if len(parts) > 2 else parts[1]
        else:
            # Just ingredient name
            amount = "some"
            ingredient_name = item
        
        ingredients.append({
            "name": ingredient_name.lower().strip(),
            "amount": amount
        })
    
    return ingredients