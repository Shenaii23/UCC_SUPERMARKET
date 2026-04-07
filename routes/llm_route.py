# routes/llm_route.py
import json
import time 
import pandas as pd
import asyncio
import re
from typing import List, Dict, Any
from fastapi import APIRouter, Form
from fastapi.responses import StreamingResponse
from models.llm_classes import IntentQuery, Entity
from services.user_intent_executive import fetch_intent, message_to_llm, intent_detection, llm_response_extraction
from services.llm_response import format_message, save_products, save_recipe, stream_message, extract_product_names_from_response
from services.context_handler import preprocess_message, update_user_state, get_user_state
from intents.greetings import is_greeting, handle_cart
from intents.stock_check import was_stock_check
from intents.cart_logic import add_to_cart, formatted_cart_summary
from models.inventory_management import CartUpdate



llm_router = APIRouter()

SLOW_INTENTS = {"get_recipe", "check_recipe_availability", "recommend_recipe", "budget_recipe_suggestion"}

# Its just a greeting
async def stream_greeting_message(response_text: str):
    """Stream a greeting message word by word."""
    words = response_text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'word': chunk})}\n\n"
        await asyncio.sleep(0.04)
    yield f"data: {json.dumps({'done': True})}\n\n"

def cart_redirect():
    return json.dumps({"message": "Sure! Redirecting you to your cart now.",
                       'action': 'redirect_to_cart',
                       'redirect_url': '/cart'})

async def process_and_stream(user_message: str, user_id: str):
    # Check if it's a greeting and stream the response immediately
    REORDER_KEYWORDS = ["reorder my last shopping list"]

    if any(phrase in user_message.lower() for phrase in REORDER_KEYWORDS):
        
        reponse = "Sorry, we dont have that feature yet. You can send an image of your last shopping list and I can try to add it to your cart."
        async for chunk in stream_message(reponse):
            yield chunk
        return

    greeting_response = is_greeting(user_message)
    if greeting_response and any(phrase in user_message.lower() for phrase in ["show cart", "what's in my cart?", "what is in my cart?", "what do i have in my cart?", "cart summary", "my cart", "show me my cart"]):
            greeting_response = handle_cart(user_id)
            # Stream the cart response immediately
            async for chunk in stream_greeting_message(greeting_response):
                yield chunk

            # Save the cart response to history
            state = get_user_state(user_id)
            state["raw_history"].append({"role": "user", "content": user_message})
            state["raw_history"].append({"role": "assistant", "content": greeting_response})

            return
    
    if greeting_response:
        async for chunk in stream_greeting_message(greeting_response):
            yield chunk
        # UPDATE: Save greeting to raw history
        state = get_user_state(user_id)
        state["raw_history"].append({"role": "user", "content": user_message})
        state["raw_history"].append({"role": "assistant", "content": greeting_response})
        return

    direct_cart_messages = [
    "cart summary",
    "cart-summary",
    "show cart",
    "show my cart",
    "what's in my cart",
    "what is in my cart",
    "what do i have in my cart",
    "show me my cart",
    "what's in my cart",
    "what is in my cart",
    "what do i have in my cart",
    "view cart",
    "list my cart",
    "check cart"]

    # Recipe intent

    recipe_intent_keywords = ["make", "cook", "recipe", "prepare", "suggestion", "ideas", "dinner", "lunch", "breakfast", "meal"]
    user_msg_lower = user_message.lower()

    # 3. Enhanced Logic: Check if it's a cart phrase AND NOT a recipe request
    is_cart_phrase = any(phrase in user_msg_lower for phrase in direct_cart_messages)
    is_recipe_request = any(word in user_msg_lower for word in recipe_intent_keywords)
    print(f"User Message: {user_message}")
    print(f"Is Cart Phrase: {is_cart_phrase} | Is Recipe Request: {is_recipe_request}")
    if is_cart_phrase and not is_recipe_request:
        print("Direct cart access detected. Streaming cart summary immediately.")
        direct_cart_response = formatted_cart_summary(user_id)

        async for chunk in stream_message("Hmm, let me check..."):
            if '"done": true' in chunk or '"done": True' in chunk:
                continue
            yield chunk

        yield f"data: {json.dumps({'word': '\n\n'})}\n\n"

        async for chunk in stream_message(direct_cart_response):
            yield chunk

        # NEW: Checkout button trigger for direct cart path
        yield f"data: {json.dumps({'show_checkout': True})}\n\n"

        state = get_user_state(user_id)
        state["raw_history"].append({"role": "user", "content": user_message})
        state["raw_history"].append({"role": "assistant", "content": direct_cart_response})
        return

    # For user experience show a message to the user that the LLM is thinking one line at a time, while we do the processing and intent detection
    async for chunk in stream_message("Hmm, let me check..."):
        if '"done": true' in chunk or '"done": True' in chunk:
            continue
        yield chunk
    
    # Send a double newline so the real answer starts on a new paragraph
    yield f"data: {json.dumps({'word': '\\n\\n'})}\\n\\n"

    # Step 1: Preprocess for corrections/follow-ups
    state = get_user_state(user_id)
    pending_selection = state.get("pending_selection", [])

    print(f"Pending selection before processing: {pending_selection}")
    print(f"User State: {state}")
    if pending_selection:
        normalized = user_message.lower().strip()
        yes_responses = {"yes", "y", "yep", "yeah", "yup", "sure", "ok", "okay", "affirmative", "right", "sounds good"}
        no_responses = {"no", "nope", "nah", "negative", "not really", "never", "dont", "don't"}
        if normalized in yes_responses:
            user_message = f"Add {', '.join(pending_selection)} to my cart"
            state["pending_selection"] = []
        elif normalized in no_responses:
            response = "Okay! I won't add that item to your cart. Let me know if you'd like something else."
            state["history"].append({"role": "user", "content": user_message})
            state["history"].append({"role": "assistant", "content": response})
            state["pending_selection"] = []
            async for chunk in stream_message(response):
                yield chunk
            return
        
    if user_message.lower() in ["take me to my cart", "i want to see my cart", "show me my cart", "take me to my basket", "i want to see my basket"]:
        redirect_response = "Sure! Redirecting you to your cart now."
        async for chunk in stream_message(redirect_response):
            yield chunk
        redirect_cart = cart_redirect()
        return

    # Check the user's last message was stock check and got a vague response like "yes" or "no"
    stock_check_response = was_stock_check(user_message, user_id)
    if stock_check_response:
        async for chunk in stream_greeting_message(stock_check_response):
            yield chunk
        # UPDATE: Save to history
        state = get_user_state(user_id)
        state["history"].append({"role": "user", "content": user_message})
        state["history"].append({"role": "assistant", "content": stock_check_response})
        return

    state = get_user_state(user_id)
    chat_history = state.get("history", [])
    processed_message = preprocess_message(user_id, user_message)
    
    # Step 2: Detect intent
    intent_raw = await message_to_llm(fetch_intent(), processed_message, chat_history)
    #print("LLM intent response:", intent_raw)
    
    # Step 3: Extract intent early for holding message
    execution       = llm_response_extraction(intent_raw)
    detected_intent = execution.intent
    entities        = execution.entities
    #print(f"Intent: {detected_intent} | Entities: {entities}")
    
    # Step 4: Stream holding message for slow intents immediately
    if detected_intent in SLOW_INTENTS:
        from services.llm_response import get_holding_message
        holding = get_holding_message(detected_intent)
        for i, word in enumerate(holding.split(" ")):
            chunk = word + (" " if i < len(holding.split(" ")) - 1 else "")
            yield f"data: {json.dumps({'word': chunk})}\n\n"
            await asyncio.sleep(0.04)
        yield f"data: {json.dumps({'holding_done': True})}\n\n"
        await asyncio.sleep(0.1)
        yield f"data: {json.dumps({'clear': True})}\n\n"
        await asyncio.sleep(0.1)
    
    # Step 5: Execute full intent logic
    intent_result = await intent_detection(intent_raw, processed_message, user_id, chat_history)
    print("Intent Result:", intent_result)
    print("Processed Message:", processed_message)
    print("Chat History:", chat_history)
    
    # Step 6: Parse result
    try:
        parsed = json.loads(intent_result) if isinstance(intent_result, str) else intent_result
    except Exception:
        parsed = {"message": str(intent_result)}
    
    # Handle None case
    if parsed is None:
        parsed = {"message": ""}

    # Send structured cart additions to frontend for dedicated rendering
    if parsed.get("added_products"):
        yield f"data: {json.dumps({'added_products': parsed['added_products']})}\n\n"
    
    # NEW: Send recipe suggestions for UI rendering
    if parsed.get("suggestions") or parsed.get("recommendations"):
        recipe_options = parsed.get("suggestions") or parsed.get("recommendations")
        yield f"data: {json.dumps({'suggestions': recipe_options})}\n\n"
    
    # STEP 6.5: Contextualize the intent
    actual_intent = get_user_state(user_id).get("last_intent", detected_intent)

    # NEW: Checkout button trigger
    # Trigger if intent is checkout/view_cart OR if the response text contains a cart total
    if actual_intent in ["checkout", "view_cart"] or "Total:" in str(parsed.get("message", "")):
        yield f"data: {json.dumps({'show_checkout': True})}\n\n"

    # STEP 6.6: Extract and save products from search responses (stock_check OR inventory_check)
    if actual_intent in ["stock_check", "inventory_check", "follow_up"]:
        extracted_products = extract_product_names_from_response(parsed.get("message", ""), parsed)
        if actual_intent == "inventory_check" and not extracted_products and parsed.get("products"):
             extracted_products = parsed["products"] # Backup for inventory_check structure
             
        if extracted_products:
            save_products(user_id, extracted_products)
            # Emit product_suggestions for UI rendering (limited to top 30)
            yield f"data: {json.dumps({'product_suggestions': extracted_products[:30]})}\n\n"
            #print(f"[Extract] Found {len(extracted_products)} products: {[p['product_name'] for p in extracted_products]}")
    
    # Step 7: Save suggestions for context detection
    suggestions = []
    raw_suggestions = (parsed.get("recommendations") or parsed.get("suggestions") or [])
    for r in raw_suggestions:
        if isinstance(r, str):
            suggestions.append(r)
        elif isinstance(r, dict):
            suggestions.append(r.get("recipe_name", ""))
    
    # Filter empty strings and duplicates
    suggestions = [s for s in suggestions if s]
    
    # Format the response for display
    message_text = format_message(parsed)
    
    # Strip any HTML tags and entities that might have been generated by the LLM
    import re
    import html
    message_text = re.sub(r'<[^>]+>', '', message_text)  # Remove HTML tags
    message_text = html.unescape(message_text)  # Convert HTML entities to plain text
    
    # This ensures the next request has proper context
    state["history"].append({"role": "user", "content": user_message})  # ORIGINAL user input
    state["history"].append({"role": "assistant", "content": message_text})  # FORMATTED response
    
    # Update all other state info
    # Note: state["last_intent"] is now updated directly inside intent_detection() after all overrides
    state["last_entities"]     = entities.dict() if hasattr(entities, "dict") else {}
    state["last_user_message"] = user_message
    state["last_response"]     = message_text
    state["last_suggestions"]  = suggestions
    
    # NEW: Ensure recipe_name is explicitly saved to state if found in entities
    if hasattr(entities, "recipe_name") and entities.recipe_name:
        state["last_recipe_name"] = entities.recipe_name
        print(f"📄 [STATE] Saved recipe_name: {entities.recipe_name}")
    
    # Step 9: Save to session (cart, products, recipes)
    if parsed.get("added_products"):
        update_model = CartUpdate(user_id=user_id, items=parsed["added_products"])
        await add_to_cart(update_model)
        state["pending_selection"] = []
    if parsed.get("removed_products"):
        from intents.cart_logic import remove_from_cart
        for p in parsed["removed_products"]:
            remove_from_cart(user_id, product_name=p.get("product_name"))
    if parsed.get("products"):
        save_products(user_id, parsed["products"])
    if parsed.get("ingredients") or parsed.get("steps"):
        print(f"📄 SAVING RECIPE FOR {user_id}: {parsed.get('recipe_name', 'Unknown')}")
        save_recipe(user_id, parsed)
        # ╔════════════════════════════════════════════════════════════╗
        # ║  SYNC: Ingredients to state for next-turn add_to_cart      ║
        # ╚════════════════════════════════════════════════════════════╝
        if parsed.get("ingredients"):
            ing_list = []
            for ing in parsed["ingredients"]:
                if isinstance(ing, dict):
                    name = ing.get("name", "")
                    amt = ing.get("amount", "")
                    if name:
                        ing_list.append(f"{name} ({amt})" if amt else name)
                elif isinstance(ing, str):
                    ing_list.append(ing)
            
            if ing_list:
                state["last_recipe_ingredients"] = " | ".join(ing_list)
                state["last_recipe_name"] = parsed.get("recipe_name") or state.get("last_recipe_name")
                print(f"📌 SYNCED {len(ing_list)} ingredients to state: {state['last_recipe_ingredients'][:100]}...")
    if parsed.get("suggestions"):
        save_recipe(user_id, {"type": "suggestions", "data": parsed["suggestions"]})
    if parsed.get("recommendations"):
        save_recipe(user_id, {"type": "recommendations", "data": parsed["recommendations"]})
    
    # Step 10: Stream the formatted response
    words = message_text.split(" ")
    for i, word in enumerate(words):
        chunk = word + (" " if i < len(words) - 1 else "")
        yield f"data: {json.dumps({'word': chunk})}\n\n"
        await asyncio.sleep(0.04)
    yield f"data: {json.dumps({'done': True})}\n\n"


@llm_router.post("/chat")
async def detect_intent(query: IntentQuery):
    user_message = query.message.lower()
    user_id      = query.user_id or "default"


    MAX_WORDS = 100
    word_count = len(user_message.split(" "))
    
    if word_count > MAX_WORDS:
        return StreamingResponse(
            stream_greeting_message(f"Your message is too long! Please keep it under {MAX_WORDS} words. You sent {word_count} words."),
            media_type="text/event-stream"
        )   
     # Save user previous message, is in a csv file for now
    if user_message:
        import os
        import csv
        # Check if the csv file exists, if not create it and write the header
        csv_file = "user_messages.csv"
    
        # Create header if file doesn't exist
        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(["user_id", "user_message", "timestamp"])
        
        # Append message
        with open(csv_file, 'a', newline='') as f:
            csv.writer(f).writerow([user_id, user_message, int(time.time())])

    return StreamingResponse(
        process_and_stream(user_message, user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )



