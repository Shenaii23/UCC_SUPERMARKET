"""
Docstring for intents.get_recipe
 This file will handle everything related to the intents get_recipe, and to recommend a recipe based on certain criterias
"""

from datasets.data import recipes_df
from intents.stock_check import get_inventory_data
from models.llm_classes import IntentQuery, Entity, IntentExecution, LLMResponse

# Get recipe data
def get_recipe_data(entities: Entity):
    """
    Searches recipe dataset for matching recipe names.

    :param entities: Entity object containing recipe_name
    :type entities: Entity

    Returns:
        list[dict]: [
            {
                "recipe_name": str,
                "ingredients_with_amounts": str,
                "servings": int,
                "instructions": str
            }
        ]

    Returns empty list if no matches found.
    """
    search_term = entities.recipe_name
    
    if not search_term or len(search_term) < 3:
        return []

    # Filter the DataFrame
    matches = recipes_df[
        recipes_df['recipe_name'].str.contains(search_term, case=False, na=False)
    ]
    
    # Return just the raw data
    return matches[['recipe_name', 'ingredients_with_amounts', 'servings', 'instructions']].to_dict(orient='records')