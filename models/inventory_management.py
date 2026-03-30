from pydantic import BaseModel
from typing import List, Dict


class StockQuery(BaseModel):
    codes: List[str]
    user_id: str

class RecipeQuery(BaseModel):
    recipe_code: str
    servings: int
    user_id: str

class CartUpdate(BaseModel):
    user_id: str
    items: List[Dict]