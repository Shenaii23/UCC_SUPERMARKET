from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from routes.main_routes import main_router
from routes.api_routes import cart_router
from routes.llm_route import llm_router
from shoppingList.shopping_list import shoplist_router

app = FastAPI()

# Register the router
app.include_router(main_router)
# Regster the api route
app.include_router(cart_router)
# Register the llm route
app.include_router(llm_router)
# Register the shopping list route
app.include_router(shoplist_router)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

if __name__ == "__main__":
    print("🚀 Starting UCC Supermarket Chat Assistant...")
    print("📦 Loading routes and middleware...")
    import uvicorn
    print("🌐 Starting FastAPI server on http://127.0.0.1:8004")
    print("💡 Press Ctrl+C to stop the server")
    uvicorn.run(app, host="127.0.0.1", port=8004, log_level="info")