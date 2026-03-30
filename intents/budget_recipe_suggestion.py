"""
budget_recipe_suggestion.py
"""

from datasets.data import recipes_df, inventory_df
from models.llm_classes import Entity
from intents.check_recipe_availability import extract_ingredients
from intents.stock_check import get_inventory_data


def _build_inventory_cache() -> dict:
    """
    Pre-loads all inventory data into a dict keyed by ingredient name (lowercase).
    One DataFrame scan instead of one per ingredient per recipe.
    """
    cache = {}
    for _, row in inventory_df.iterrows():
        key = row["name"].strip().lower()
        cache.setdefault(key, []).append(row.to_dict())
    return cache


# Module-level cache — built once when the module loads
_INVENTORY_CACHE = _build_inventory_cache()


def _get_cached(name: str) -> list:
    """Fuzzy-ish lookup: tries exact match, then checks if any key contains the term."""
    key = name.strip().lower()
    if key in _INVENTORY_CACHE:
        return _INVENTORY_CACHE[key]
    # Fallback: partial match (mirrors what get_inventory_data likely does)
    return [v for k, vals in _INVENTORY_CACHE.items() if key in k for v in vals]


def _analyse_ingredients(ingredients: list) -> tuple[float, list[str]]:
    """
    Single pass over ingredients — returns (estimated_cost, missing_names).
    Replaces the two separate loops that each called get_inventory_data.
    """
    total = 0.0
    missing = []

    for ingredient in ingredients:
        matches = _get_cached(ingredient["name"])
        if not matches:
            missing.append(ingredient["name"])
            continue
        total += min(matches, key=lambda x: x["price"])["price"]

    return round(total, 2), missing


def budget_recipe_suggestion(entities: Entity) -> dict:
    budget = entities.budget

    if not budget:
        return {"error": "No budget provided."}

    suggestions = []

    for _, row in recipes_df.iterrows():
        ingredients = extract_ingredients(row["ingredients_with_amounts"])
        if not ingredients:
            continue

        cost, missing = _analyse_ingredients(ingredients)  # one pass, not two

        if cost <= budget:
            suggestions.append({
                "recipe_name":         row["recipe_name"],
                "category":            row.get("category", ""),
                "servings":            row.get("servings", ""),
                "estimated_cost":      cost,
                "missing_ingredients": missing,
                "instructions":        row["instructions"],
            })

    suggestions.sort(key=lambda x: x["estimated_cost"])

    return {
        "budget":      budget,
        "suggestions": suggestions[:5],
        "error":       None,
    }