from fastapi import Request, APIRouter
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datasets.data import inventory_df, recipes_df, carts
from models.inventory_management import StockQuery, RecipeQuery, CartUpdate


templates = Jinja2Templates(directory="templates")

main_router = APIRouter()

@main_router.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@main_router.get("/products", response_class=HTMLResponse)
def products(request: Request):
    return templates.TemplateResponse("products.html", {
        "request": request,
        "products": inventory_df.to_dict(orient="records")
    })

@main_router.get("/cart", response_class=HTMLResponse)
def cart(request: Request, user_id: str = "testuser"):
    user_cart = carts.get(user_id, [])
    return templates.TemplateResponse("cart.html", {
        "request": request,
        "cart": user_cart
    })

@main_router.get("/debug/carts")
def debug_carts():
    """Diagnostic route to check shared memory state"""
    return {
        "carts_ids": id(carts),
        "total_carts": len(carts),
        "all_data": carts
    }