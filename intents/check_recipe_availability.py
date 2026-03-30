"""
check_recipe_availability.py
User wants to know if we have the ingredients for a recipe in stock.
Example: "Can I make Ackee and Saltfish with what you have?"
"""

from datasets.data import recipes_df
from models.llm_classes import Entity
from intents.stock_check import get_inventory_data
from intents.get_recipe import get_recipe_data


def extract_ingredients(ingredient_str: str) -> list:
    """Parse 'Ackee (1 can) | Saltfish (250g)' into [{'name': 'Ackee', 'amount': '1 can'}, ...]"""
    if not ingredient_str:
        return []
    result = []
    for part in ingredient_str.split("|"):
        part = part.strip()
        name   = part.split("(")[0].strip()
        amount = part.split("(")[1].split(")")[0] if "(" in part and ")" in part else None
        result.append({"name": name, "amount": amount})
    return result


def check_recipe_availability(entities: Entity) -> dict:
    """
    Checks if the ingredients for a recipe are available in inventory.

    Returns:
        {
            "recipe_name": str,
            "available":   [{"name": str, "amount": str, "products": list}],
            "unavailable": [{"name": str, "amount": str}],
            "fully_available": bool,
            "error": str | None
        }
    """
    recipe_data = get_recipe_data(entities)

    if not recipe_data:
        return {"error": f"Sorry, I couldn't find a recipe for '{entities.recipe_name}'."}

    recipe      = recipe_data[0]
    ingredients = extract_ingredients(recipe["ingredients_with_amounts"])

    if not ingredients:
        return {"error": "No ingredients listed for this recipe."}

    report = {
        "recipe_name":    recipe["recipe_name"],
        "available":      [],
        "unavailable":    [],
        "fully_available": False,
        "error":          None
    }

    for ingredient in ingredients:
        inventory = get_inventory_data(ingredient["name"])
        in_stock  = [i for i in inventory if int(i.get("quantity", 0)) > 0]

        if in_stock:
            report["available"].append({
                "name":     ingredient["name"],
                "amount":   ingredient["amount"],
                "products": in_stock
            })
        else:
            report["unavailable"].append({
                "name":   ingredient["name"],
                "amount": ingredient["amount"]
            })

    report["fully_available"] = len(report["unavailable"]) == 0
    return report