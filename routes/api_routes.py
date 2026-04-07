from datasets.data import inventory_df, sales_df, carts
from models.inventory_management import StockQuery, RecipeQuery, CartUpdate
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import json

cart_router = APIRouter()

class CartItem(BaseModel):
    product_code: Optional[str] = None
    quantity: int = 1
    product_name: Optional[str] = None
    price: Optional[float] = None

class CartUpdateRequest(BaseModel):
    user_id: str
    items: List[CartItem]

class QuantityUpdate(BaseModel):
    user_id: str
    product_code: str
    quantity: int  # +1 or -1 for increment/decrement

class RemoveItem(BaseModel):
    user_id: str
    product_code: str

class ClearCartRequest(BaseModel):
    user_id: str

# Add to cart
@cart_router.post("/add_to_cart")
async def add_to_cart(update: CartUpdateRequest):
    """Add items to user's cart"""
    user_cart = carts.get(update.user_id, [])
    
    for item in update.items:
        # Check if item already exists in cart
        existing_item = None
        for cart_item in user_cart:
            if cart_item["product_code"] == item.product_code:
                existing_item = cart_item
                break
        
        if existing_item:
            # Update quantity if already in cart
            existing_item["quantity"] += item.quantity
        else:
            # Get product details from inventory if not provided
            product = inventory_df[inventory_df['product_code'] == item.product_code]
            if not product.empty:
                product = product.iloc[0]
                cart_item = {
                    "product_code": str(item.product_code),
                    "product_name": str(product['product_name']),
                    "price": float(product['price']) if hasattr(product['price'], 'item') else float(product['price']),
                    "quantity": item.quantity,
                    "category": str(product.get('category', ''))
                }
                user_cart.append(cart_item)
            else:
                # If product not found but we have details from frontend
                if item.product_name and item.price:
                    cart_item = {
                        "product_code": str(item.product_code),
                        "product_name": item.product_name,
                        "price": item.price,
                        "quantity": item.quantity,
                        "category": ""
                    }
                    user_cart.append(cart_item)
    
    carts[update.user_id] = user_cart
    return {"cart": user_cart, "success": True}

# Remove from cart
@cart_router.post("/remove_from_cart")
async def remove_from_cart(remove: RemoveItem):
    """Remove an item completely from cart"""
    user_cart = carts.get(remove.user_id, [])
    
    # Filter out the item to remove
    updated_cart = [item for item in user_cart if item["product_code"] != remove.product_code]
    
    carts[remove.user_id] = updated_cart
    return {"cart": updated_cart, "success": True}

# Update cart quantity (increment/decrement)
@cart_router.post("/update_cart")
async def update_cart_quantity(update: QuantityUpdate):
    """Update quantity of an item in cart (+1 or -1)"""
    user_cart = carts.get(update.user_id, [])
    
    for i, item in enumerate(user_cart):
        if item["product_code"] == update.product_code:
            new_quantity = item["quantity"] + update.quantity
            if new_quantity <= 0:
                # Remove item if quantity becomes 0 or negative
                user_cart.pop(i)
            else:
                item["quantity"] = new_quantity
            break
    
    carts[update.user_id] = user_cart
    return {"cart": user_cart, "success": True}

# Clear cart
@cart_router.post("/clear_cart")
async def clear_cart(clear: ClearCartRequest):
    """Clear all items from user's cart"""
    carts[clear.user_id] = []
    return {"cart": [], "success": True, "message": "Cart cleared successfully"}

# Get cart
@cart_router.get("/get_cart")
async def get_cart(user_id: str):
    """Get user's current cart"""
    user_cart = carts.get(user_id, [])
    return {"cart": user_cart, "success": True}

# Get cart via POST (for compatibility)
@cart_router.post("/get_cart")
async def get_cart_post(request: dict):
    """Get user's current cart (POST method for compatibility)"""
    user_id = request.get("user_id", "default")
    user_cart = carts.get(user_id, [])
    return {"cart": user_cart, "success": True}

# Optional: Get all products endpoint
@cart_router.get("/api/products")
async def get_products(user_id: str = "default"):
    """Get all available products"""
    products = []
    for _, row in inventory_df.iterrows():
        products.append({
            "product_code": str(row['product_code']),
            "product_name": str(row['product_name']),
            "category": str(row.get('category', 'General')),
            "price": float(row['price']) if hasattr(row['price'], 'item') else float(row['price']),
            "sale_unit": str(row.get('sale_unit', 'each'))
        })
    return {"products": products}

# Optional: Get single product
@cart_router.get("/product/{product_code}")
async def get_product(product_code: str):
    """Get details for a specific product"""
    product = inventory_df[inventory_df['product_code'] == product_code]
    if product.empty:
        raise HTTPException(status_code=404, detail="Product not found")
    
    row = product.iloc[0]
    return {
        "product_code": str(row['product_code']),
        "product_name": str(row['product_name']),
        "category": str(row.get('category', 'General')),
        "price": float(row['price']) if hasattr(row['price'], 'item') else float(row['price']),
        "sale_unit": str(row.get('sale_unit', 'each'))
    }