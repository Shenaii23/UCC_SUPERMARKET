# services/llm_response.py
import json
import asyncio
import re

# Server-side session store — keyed by user_id
product_sessions: dict = {}
recipe_sessions:  dict = {}


def extract_product_names_from_response(message_text: str, parsed: dict) -> list:
    """
    Extract product names from both the LLM response and the parsed JSON.
    Returns a list of dicts: [{"product_name": "...", "price": ..., "quantity": ...}]
    """
    products = []
    
    # First, try to get from parsed JSON (best case)
    if parsed.get("products"):
        return parsed["products"]
    
    # Otherwise, try to extract from message text using regex
    # Pattern: "Product Name ($price)" or just "Product Name"
    # Examples: "Stock - Chicken ($120)", "Boneless Skinless Chicken Breast ($600)"
    
    pattern = r"([A-Za-z\s\-]+?)\s*\(\$(\d+(?:,\d+)*)\)"
    matches = re.findall(pattern, message_text)
    
    for product_name, price in matches:
        products.append({
            "product_name": product_name.strip(),
            "price": int(price.replace(",", "")),
            "quantity": 1  # Default quantity
        })
    
    return products



def save_products(user_id: str, products: list):
    """
    Save product list to session.
    Products should be a list of dicts with keys: product_name, price, quantity
    """
    # Ensure products have the right structure
    cleaned = []
    for p in products:
        if isinstance(p, dict):
            cleaned.append({
                "product_name": p.get("product_name", ""),
                "price": p.get("price", 0),
                "quantity": p.get("quantity", 1)
            })
    
    product_sessions[user_id] = cleaned


def get_products(user_id: str) -> list:
    """Get saved products from session."""
    return product_sessions.get(user_id, [])


def save_recipe(user_id: str, recipe: dict):
    recipe_sessions[user_id] = recipe


def get_recipe(user_id: str) -> dict:
    return recipe_sessions.get(user_id, {})


def format_message(parsed: dict) -> str:
    """
    Builds a rich formatted message from the parsed LLM JSON
    based on what keys are present.
    """
    message = parsed.get("message", "")
    intent_type = detect_shape(parsed)

    if intent_type == "recipe":
        ingredients = parsed.get("ingredients", [])
        steps       = parsed.get("steps", [])
        servings    = parsed.get("servings", "")

        lines = [message, ""]

        if servings:
            lines.append(f"Serves: {servings}\n")

        if ingredients:
            lines.append("Ingredients:")
            for ing in ingredients:
                lines.append(f"  • {ing['name']} — {ing['amount']}")
            lines.append("")

        if steps:
            lines.append("Instructions:")
            for step in steps:
                lines.append(f"  {step}")

        return "\n".join(lines)

    elif intent_type == "availability":
        available   = parsed.get("available", [])
        unavailable = parsed.get("unavailable", [])
        lines       = [message, ""]

        if available:
            lines.append("✅ In stock:")
            for item in available:
                lines.append(f"  • {item}")
            lines.append("")

        if unavailable:
            lines.append("❌ Not available:")
            for item in unavailable:
                lines.append(f"  • {item}")

        return "\n".join(lines)

    elif intent_type == "suggestions":
        suggestions = parsed.get("suggestions", []) or parsed.get("recommendations", [])
        lines       = [message, ""]

        for s in suggestions:
            if isinstance(s, str):
                name = s
                detail = ""
            else:
                name    = s.get("recipe_name", "")
                cost    = s.get("estimated_cost", "")
                score   = s.get("match_score", "")
                serves  = s.get("servings", "")

                detail = ""
                if cost:
                    detail += f"Est. cost: ${cost}"
                if score:
                    detail += f"  Match: {score}%"
                if serves:
                    detail += f"  Serves: {serves}"

            lines.append(f"  • {name}" + (f" — {detail}" if detail else ""))

        return "\n".join(lines)

    # Default — just return the message as-is
    return message


def detect_shape(parsed: dict) -> str:
    """Detect what kind of response this is based on keys present."""
    if "ingredients" in parsed or "steps" in parsed:
        return "recipe"
    if "available" in parsed or "unavailable" in parsed:
        return "availability"
    if "suggestions" in parsed or "recommendations" in parsed:
        return "suggestions"
    return "message"


async def stream_message(message: str):
    """Stream a message word by word as SSE events."""
    words = message.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'word': chunk})}\n\n"
        await asyncio.sleep(0.04)
    yield f"data: {json.dumps({'done': True})}\n\n"


# ─────────────────────────────────────────────
#  HOLDING MESSAGES PER INTENT
# ─────────────────────────────────────────────

import random

HOLDING_MESSAGES = {
    "get_recipe": [
        "Let me pull up that recipe for you...",
        "One moment, finding that recipe...",
        "Looking that up for you right now...",
    ],
    "check_recipe_availability": [
        "Checking our shelves for those ingredients...",
        "Let me see what we have in stock for that...",
        "Scanning the inventory for that recipe...",
    ],
    "recommend_recipe": [
        "Let me think about what you could make with that...",
        "Finding some recipe ideas for you...",
        "Searching for recipes based on what you mentioned...",
    ],
    "budget_recipe_suggestion": [
        "Let me see what we can make with your budget...",
        "Checking recipes within your price range...",
        "Finding the best options for your budget...",
    ],
    "add_to_cart": [
        "Adding that to your cart...",
        "On it! Adding to your cart...",
    ],
    "general_chat": [
        "One moment...",
    ],
}


def get_holding_message(intent: str) -> str:
    options = HOLDING_MESSAGES.get(intent, ["One moment..."])
    return random.choice(options)


async def stream_holding_then_response(intent: str, response_message: str):
    """
    Streams a holding message immediately, waits for LLM,
    then clears and streams the real response — all in one SSE connection.
    """
    holding = get_holding_message(intent)

    # Stream holding message word by word
    words = holding.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'word': chunk})}\n\n"
        await asyncio.sleep(0.04)

    # Tell frontend holding is done
    yield f"data: {json.dumps({'holding_done': True})}\n\n"
    await asyncio.sleep(0.1)

    # Tell frontend to clear the bubble and start fresh
    yield f"data: {json.dumps({'clear': True})}\n\n"
    await asyncio.sleep(0.1)

    # Stream the real response
    words = response_message.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'word': chunk})}\n\n"
        await asyncio.sleep(0.04)

    yield f"data: {json.dumps({'done': True})}\n\n"