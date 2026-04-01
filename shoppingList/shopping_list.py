"""
shoppingList/shopping_list.py
Process shopping list images and confirm matches with user before adding to cart
"""
import re
import json
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
import io
import pytesseract
from PIL import Image
from fastapi import File, UploadFile, Form, APIRouter
import google.generativeai as genai
import pandas as pd

shoplist_router = APIRouter()

load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

# Configure tesseract (remove this for Render - will use system tesseract)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Users\sheri\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'


@shoplist_router.post("/process_shopping_image")
async def process_shopping_image(
    user_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Process uploaded image, extract items, and ask user to confirm matches.
    Does NOT add to cart automatically - requires confirmation.
    """
    try:
        # Read image
        img = Image.open(image.file)
        
        # Extract items from image using Gemini
        prompt = """
        This is a photo of a shopping list (handwritten or typed).
        1. Transcribe all items listed
        2. Extract quantities (default to 1 if not specified)
        3. Return ONLY a JSON array, no other text:
        [{"product_name": "item name", "quantity": number}]
        """
        response = model.generate_content(
            [prompt, img],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Parse extracted items
        extracted_items = json.loads(response.text)
        
        # Find potential matches for each item
        items_with_matches = []
        for item in extracted_items:
            matches = await find_product_matches(item["product_name"], limit=3)
            items_with_matches.append({
                "requested_name": item["product_name"],
                "quantity": item["quantity"],
                "potential_matches": matches
            })
        
        return {
            "success": True,
            "extracted_items": items_with_matches,
            "message": "I found these items in your shopping list. Please confirm which products you'd like to add to your cart."
        }
        
    except Exception as e:
        import traceback
        print(f"Error processing image:\n{traceback.format_exc()}")
        return {
            "success": False,
            "error": f"Error processing image: {str(e)}"
        }


@shoplist_router.post("/confirm_shopping_items")
async def confirm_shopping_items(
    user_id: str = Form(...),
    items_json: str = Form(...)
):
    """
    User confirms which products to add from the matched options.
    
    Expected format:
    {
        "items": [
            {
                "requested_name": "flour",
                "selected_product": "All Purpose Flour",
                "quantity": 2
            },
            {
                "requested_name": "milk",
                "selected_product": "Whole Milk",
                "quantity": 1
            }
        ]
    }
    """
    try:
        confirmed_items = json.loads(items_json)
        items_list = confirmed_items.get("items", [])
        
        # Add confirmed items to cart
        result = await add_confirmed_items_to_cart(user_id, items_list)
        
        return {
            "success": True,
            "added": result["added"],
            "failed": result["failed"],
            "message": f"Added {result['total_added']} items to your cart!"
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


async def find_product_matches(product_name: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Find potential matches for a product name from inventory.
    Returns top matches with exact and fuzzy matching.
    """
    try:
        df = pd.read_csv('datasets/inventory_updated.csv')
        product_lower = product_name.lower()
        
        # Pass 1: Exact matches
        exact = df[df['product_name'].str.lower() == product_lower]
        if not exact.empty:
            match = exact.iloc[0]
            return [{
                "product_name": match['product_name'],
                "category": match['category'],
                "price": float(match['price']),
                "in_stock": int(match['quantity']) > 0,
                "confidence": "exact"
            }]
        
        # Pass 2: Keyword matches (product_name contains any word from search)
        keywords = product_lower.split()
        keyword_matches = []
        
        for _, row in df.iterrows():
            row_name_lower = row['product_name'].lower()
            # Count how many keywords match
            matches_count = sum(1 for kw in keywords if kw in row_name_lower)
            if matches_count > 0:
                keyword_matches.append({
                    "product_name": row['product_name'],
                    "category": row['category'],
                    "price": float(row['price']),
                    "in_stock": int(row['quantity']) > 0,
                    "matches": matches_count
                })
        
        # Sort by match count and return top results
        keyword_matches.sort(key=lambda x: (-x['matches'], x['product_name']))
        
        results = []
        for match in keyword_matches[:limit]:
            match.pop("matches", None)
            match["confidence"] = "partial"
            results.append(match)
        
        return results if results else [{"product_name": product_name, "note": "No exact matches found"}]
        
    except Exception as e:
        print(f"Error finding matches for '{product_name}': {e}")
        return [{"product_name": product_name, "note": "Error searching inventory"}]


async def add_confirmed_items_to_cart(user_id: str, items: List[Dict]) -> Dict:
    """
    Add user-confirmed items to cart.
    Items already have user's selected product name.
    """
    from intents.cart_logic import add_to_cart
    from models.inventory_management import CartUpdate
    
    added = []
    failed = []
    
    for item in items:
        try:
            selected_product = item.get("selected_product", item.get("requested_name"))
            
            cart_update = CartUpdate(
                user_id=user_id,
                items=[{
                    "product_name": selected_product,
                    "quantity": item["quantity"],
                    "notes": "from shopping list"
                }]
            )
            await add_to_cart(cart_update)
            added.append(selected_product)
        except Exception as e:
            failed.append({
                "item": item.get("selected_product", item.get("requested_name")),
                "error": str(e)
            })
    
    return {
        "added": added,
        "failed": failed,
        "total_added": len(added)
    }