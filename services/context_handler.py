# services/context_handler.py
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher  # ← For fuzzy matching
import re

USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Fuzzy matching threshold (0-1, higher = stricter)
FUZZY_THRESHOLD = 0.6  # 60% match is good enough


def get_user_state(user_id: str) -> Dict[str, Any]:
    if user_id not in USER_SESSIONS:
        USER_SESSIONS[user_id] = {
            "last_intent":       None,
            "last_entities":     {},
            "last_user_message": None,
            "last_response":     None,
            "last_suggestions":  [],   # stores recipe names from last recommendation
            "history":           [],   # stores full chat history for the LLM
            "raw_history":       [],   # stores raw user messages and assistant responses for context
            "pending_selection": []
        }
    return USER_SESSIONS[user_id]


def update_user_state(user_id: str, intent: str = None, entities: dict = None, user_message: str = None, 
                      response: str = None, suggestions: list = None):
    """
    Update user session state. 
    NOTE: History is now managed directly in llm_route.py for better control.
    """
    state = get_user_state(user_id)
    
    if intent is not None:
        state["last_intent"] = intent
    if entities is not None:
        state["last_entities"] = entities
    if user_message is not None:
        state["last_user_message"] = user_message
    if response is not None:
        state["last_response"] = response
    if suggestions is not None:
        state["last_suggestions"] = suggestions


def set_pending_selection(user_id: str, selections: list):
    state = get_user_state(user_id)
    state["pending_selection"] = selections


def clear_pending_selection(user_id: str):
    state = get_user_state(user_id)
    state["pending_selection"] = []


def get_pending_selection(user_id: str) -> list:
    return get_user_state(user_id).get("pending_selection", [])


def is_confirmation_response(user_message: str) -> Optional[str]:
    text = user_message.lower().strip()
    yes = {"yes", "y", "yep", "yeah", "yup", "sure", "ok", "okay", "affirmative", "right", "sounds good"}
    no = {"no", "nope", "nah", "negative", "not really", "never", "dont", "don't"}
    if text in yes:
        return "yes"
    if text in no:
        return "no"
    return None


def fuzzy_match(user_text: str, product_name: str, threshold: float = FUZZY_THRESHOLD) -> bool:
    """
    Fuzzy match user text against product name.
    Returns True if similarity is above threshold.
    
    Examples:
    - "boneless chicken" vs "Boneless Skinless Chicken Breast" → True
    - "the first one" vs "Stock - Chicken" → False
    """
    user_text = user_text.lower().strip()
    product_name = product_name.lower().strip()
    
    # Exact substring match (best case)
    if user_text in product_name or product_name in user_text:
        return True

    # Token-level match to catch selections like "boneless chicken" -> "Boneless Skinless Chicken Breast"
    user_tokens = set(re.findall(r"\w+", user_text))
    product_tokens = set(re.findall(r"\w+", product_name))
    if user_tokens and user_tokens.issubset(product_tokens):
        return True
    
    # Fuzzy similarity score
    similarity = SequenceMatcher(None, user_text, product_name).ratio()
    return similarity >= threshold


def is_correction(user_message: str) -> bool:
    text     = user_message.lower()
    triggers = ["i said", "no,", "not that", "i meant", "wrong", "instead", "only", "dont", "do not"]
    return any(trigger in text for trigger in triggers)


def is_recipe_selection(user_message: str, state: dict) -> bool:
    """
    Detect if the user is selecting a recipe from the last recommendation list.
    Supports: exact name, fuzzy match, or ordinal selection (first, second, etc.)
    """
    if state.get("last_intent") not in ("recommend_recipe", "get_recipe"):
        return False

    text        = user_message.lower().strip()
    suggestions = state.get("last_suggestions", [])

    if not suggestions:
        return False

    # Check for exact/fuzzy match
    for recipe in suggestions:
        if fuzzy_match(text, recipe):
            return True

    # Check for ordinal selection — "the first one", "second one"
    ordinals = ["first", "second", "third", "1st", "2nd", "3rd", "the first", "the second", "the third"]
    if any(o in text for o in ordinals):
        return True

    return False


def get_selected_recipe(user_message: str, state: dict) -> str | None:
    """
    Returns the recipe name the user is referring to.
    Supports: exact name, fuzzy match, or ordinal selection.
    """
    text        = user_message.lower().strip()
    suggestions = state.get("last_suggestions", [])

    # Check for fuzzy match first (most likely case)
    for recipe in suggestions:
        if fuzzy_match(text, recipe):
            return recipe

    # Ordinal selection
    if any(o in text for o in ["first", "1st", "the first"]) and len(suggestions) >= 1:
        return suggestions[0]
    if any(o in text for o in ["second", "2nd", "the second"]) and len(suggestions) >= 2:
        return suggestions[1]
    if any(o in text for o in ["third", "3rd", "the third"]) and len(suggestions) >= 3:
        return suggestions[2]

    return None


def rewrite_with_context(user_message: str, state: dict) -> str:
    last_intent  = state.get("last_intent")
    last_entities = state.get("last_entities", {})

    if last_intent == "recommend_recipe":
        ingredients = last_entities.get("product_name")
        return (
            f"The user previously asked for recipe recommendations using: {ingredients}. "
            f"The user now says: \"{user_message}\". "
            f"Rewrite this into a clear request for recipe recommendations."
        )

    if last_intent == "stock_check":
        product = last_entities.get("product_name")
        return (
            f"The user previously asked about: {product}. "
            f"The user now says: \"{user_message}\". "
            f"Rewrite this into a clear product-related request."
        )

    return user_message


def is_product_selection(user_message: str, user_id: str) -> bool:
    """
    Detect if the user is selecting a product from the last stock check results.
    Supports: exact name, fuzzy match, or ordinal selection.
    
    Examples that should work:
    - "Boneless chicken is fine" → matches "Boneless Skinless Chicken Breast"
    - "the first one" → matches first product in list
    - "all of them" → matches all products
    """
    from services.llm_response import get_products
    state = get_user_state(user_id)
    if state.get("last_intent") != "stock_check":
        return False

    products = get_products(user_id)
    if not products:
        return False

    text = user_message.lower().strip()
    
    # Check for "all", "everything", "both"
    if any(word in text for word in ["all", "everything", "both", "each"]):
        return True

    # Check for fuzzy match with product names
    for p in products:
        if fuzzy_match(text, p['product_name']):
            return True

    # Check for ordinal selection (first, second, third, etc.)
    ordinals = ["first", "second", "third", "fourth", "fifth", 
                "1st", "2nd", "3rd", "4th", "5th", 
                "the first", "the second", "the third"]
    if any(o in text for o in ordinals):
        return True

    return False


def get_selected_products(user_message: str, user_id: str) -> list:
    """
    Returns a list of specific product names from session based on user selection.
    Supports: exact name, fuzzy match, ordinal selection, or "all".
    
    Examples:
    - "Boneless chicken is fine" → ["Boneless Skinless Chicken Breast"]
    - "the first one" → [first product]
    - "all of them" → [all products]
    """
    from services.llm_response import get_products
    products = get_products(user_id)
    text     = user_message.lower().strip()
    selected = []

    # Check "all" first
    if any(word in text for word in ["all", "everything", "both", "each"]):
        return [p['product_name'] for p in products]

    # Try fuzzy matching with all products
    for p in products:
        if fuzzy_match(text, p['product_name']):
            selected.append(p['product_name'])

    # If no fuzzy matches, try ordinal selection
    if not selected:
        ordinal_map = {
            ("first", "1st", "the first"): 0,
            ("second", "2nd", "the second"): 1,
            ("third", "3rd", "the third"): 2,
            ("fourth", "4th", "the fourth"): 3,
            ("fifth", "5th", "the fifth"): 4,
        }
        
        for ordinal_triggers, index in ordinal_map.items():
            if any(o in text for o in ordinal_triggers):
                if len(products) > index:
                    selected.append(products[index]['product_name'])
                break  # Only select the first matched ordinal

    return selected


def preprocess_message(user_id: str, user_message: str) -> str:
    state = get_user_state(user_id)

    # 1. Product selection from stock check results
    if is_product_selection(user_message, user_id):
        selected = get_selected_products(user_message, user_id)
        if selected:
            explicit_add = bool(re.search(r"\b(add|put|buy|cart|purchase)\b", user_message.lower()))
            if not explicit_add:
                state["pending_selection"] = selected
                rewritten = f"Add {', '.join(selected)} to my cart"
                #print(f"[Context] Product selection detected: {selected} → rewritten: '{rewritten}'")
                return rewritten

    # 2. Recipe selection from previous recommendations
    if is_recipe_selection(user_message, state):
        recipe = get_selected_recipe(user_message, state)
        if recipe:
            rewritten = f"Give me the full recipe for {recipe}"
            #print(f"[Context] Recipe selection detected: {recipe} → rewritten: '{rewritten}'")
            return rewritten

    # 3. Follow-up: user is correcting the previous response
    if is_correction(user_message) and state["last_intent"]:
        rewritten = rewrite_with_context(user_message, state)
        #print(f"[Context] Correction detected → rewritten: '{rewritten}'")
        return rewritten

    return user_message