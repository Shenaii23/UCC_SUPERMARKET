import json
import os
from venv import logger
from datasets.data import inventory_df, sales_df, carts
from models.inventory_management import StockQuery, RecipeQuery, CartUpdate
from routes.main_routes import main_router, templates

# Path to persistent cart storage
CARTS_FILE = "datasets/carts_persistence.json"

def load_carts():
    """Load carts from disk if they exist"""
    if os.path.exists(CARTS_FILE):
        try:
            with open(CARTS_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_carts_to_disk():
    """Save all carts to disk"""
    try:
        with open(CARTS_FILE, "w") as f:
            json.dump(carts, f)
    except Exception as e:
        logger.error(f"Failed to save carts to disk: {e}")
# Initialize carts from disk
carts.update(load_carts())

# Add to cart
async def add_to_cart(update: CartUpdate):
    user_cart = carts.get(update.user_id, [])
    
    added_items = []
    for item in update.items:
        product_code = item.get('product_code')
        product_name = item.get('product_name')
        quantity     = int(item.get('quantity', 1))
        
        matches = None
        if product_code:
            matches = inventory_df[inventory_df['product_code'] == str(product_code)]
        if (matches is None or matches.empty) and product_name:
            # 1. Try exact match (case insensitive)
            matches = inventory_df[inventory_df['product_name'].str.lower() == str(product_name).lower()]
            
            # 2. If exact fails, try "contains" match
            if matches.empty:
                matches = inventory_df[inventory_df['product_name'].str.lower().str.contains(str(product_name).lower(), regex=False)]
            
        if matches is not None and not matches.empty:
            product = matches.iloc[0]
            product_code = str(product['product_code'])
            product_name_full = str(product['product_name'])
            price = float(product['price'])
            
            # Check if product is already in cart
            existing_item = next((i for i in user_cart if i['product_code'] == product_code), None)
            
            if existing_item:
                existing_item['quantity'] += quantity
            else:
                cart_item = {
                    "product_code": product_code,
                    "product_name": product_name_full,
                    "price": price,
                    "quantity": quantity,
                    "category": str(product.get('category', ''))
                }
                user_cart.append(cart_item)
            
            added_items.append(product_name_full)
    
    carts[update.user_id] = user_cart
    save_carts_to_disk()
    return {"cart": user_cart, "added": added_items}

# Remove from cart
def remove_from_cart(user_id: str, product_name: str | None = None, product_code: str | None = None):
    user_cart = carts.get(user_id, [])
    
    if product_code:
        new_cart = [i for i in user_cart if i['product_code'] != str(product_code)]
    elif product_name:
        new_cart = [i for i in user_cart if i['product_name'].lower() != product_name.lower()]
    else:
        return {"error": "No product identifier provided", "cart": user_cart}
    
    carts[user_id] = new_cart
    save_carts_to_disk()
    return {"message": f"Removed product from cart", "cart": new_cart}

# Get cart summary
def cart_summary(user_id: str):
    user_cart = carts.get(user_id, [])
    if not user_cart:
        return "Your cart is currently empty."
    
    total = sum(item['price'] * item['quantity'] for item in user_cart)
    summary = "Here is your current cart:\n"
    for item in user_cart:
        summary += f"- {item['product_name']} (x{item['quantity']}): ${item['price'] * item['quantity']:.2f}\n"
    
    summary += f"\nTotal: ${total:.2f}"
    return summary


def formatted_cart_summary(user_id: str) -> str:
    user_cart = carts.get(user_id, [])
    if not user_cart:
        return "Your cart is empty right now. Would you like to add anything else?"

    total = sum(item['price'] * item['quantity'] for item in user_cart)
    lines = ["Here's what's in your cart:", ""]
    for item in user_cart:
        lines.append(f"    {item['product_name']} (x{item['quantity']}): ${item['price'] * item['quantity']:.2f}")
        lines.append("")

    lines.append(f"Total: ${total:.2f}")
    lines.append("")
    lines.append("Are you ready to checkout, or would you like to add anything else to your order?")
    return "\n".join(lines)

# Clear cart
def clear_cart(user_id: str):
    carts[user_id] = []
    save_carts_to_disk()
    return {"message": "Cart cleared"}

# Get cart items
def get_cart_items(user_id: str):
    print(f"Getting cart items for user_id: {user_id}")
    return carts.get(user_id, [])


# Take me to my cart handler

from fastapi.responses import RedirectResponse

def take_me_to_cart():
    """
    Redirect the user to the cart page.
    """
    return RedirectResponse(url="/cart")