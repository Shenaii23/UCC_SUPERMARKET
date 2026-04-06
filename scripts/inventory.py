import sys
import os
# Add the parent directory to the Python path so we can import from the main project
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from json.tool import main

import pandas as pd
file = 'datasets/inventory_updated.csv'
inventory_df = pd.read_csv(file)
from pydantic import BaseModel
from typing import List, Dict
from intents.stock_check import (get_inventory_data, extract_category_from_query)

import pandas as pd
from fuzzywuzzy import process, fuzz
from nltk.stem import PorterStemmer
import nltk


class Entity(BaseModel):
    product_name: str
    category_hint: str

# Initialize the stemmer
stemmer = PorterStemmer()

def get_proactive_matches(search_term, inventory_df, threshold=70):
    """
    Finds products using stemming, fuzzy matching, and space-insensitive matching.
    """
    # 1. Normalize the search term
    search_term = search_term.lower().strip()
    
    # 2. Remove spaces for space-insensitive matching (e.g., "oxtail" vs "ox tail")
    search_term_no_spaces = search_term.replace(" ", "")
    
    # 3. Stem the search term (handles plurals: oxtails -> oxtail)
    stemmed_search = stemmer.stem(search_term)
    
    # 4. Get all unique product names
    all_products = inventory_df['product_name'].tolist()
    
    # 5. Create normalized versions of product names for matching
    def normalize_product(name):
        name_lower = name.lower()
        return {
            'original': name,
            'no_spaces': name_lower.replace(" ", ""),
            'stemmed': stemmer.stem(name_lower)
        }
    
    normalized_products = [normalize_product(p) for p in all_products]
    
    # 6. Layer 1: Exact stem match (catches "oxtails" -> "oxtail")
    stem_matches = []
    for prod in normalized_products:
        if stemmed_search == prod['stemmed']:
            stem_matches.append(prod['original'])
        # Also check if stemmed search is contained in product stem
        elif stemmed_search in prod['stemmed']:
            stem_matches.append(prod['original'])
    
    if stem_matches:
        return inventory_df[inventory_df['product_name'].isin(stem_matches)].to_dict('records')
    
    # 7. Layer 2: Space-insensitive match (catches "ox tail" -> "oxtail")
    space_matches = []
    for prod in normalized_products:
        if search_term_no_spaces == prod['no_spaces']:
            space_matches.append(prod['original'])
        elif search_term_no_spaces in prod['no_spaces']:
            space_matches.append(prod['original'])
    
    if space_matches:
        return inventory_df[inventory_df['product_name'].isin(space_matches)].to_dict('records')
    
    # 8. Layer 3: Substring match on stemmed versions
    substring_matches = inventory_df[
        inventory_df['product_name'].str.lower().str.contains(stemmed_search)
    ]
    
    if not substring_matches.empty:
        return substring_matches.to_dict('records')
    
    # 9. Layer 4: Fuzzy matching (catches typos)
    fuzzy_results = process.extract(search_term, all_products, limit=5, scorer=fuzz.token_set_ratio)
    best_matches = [name for name, score in fuzzy_results if score >= threshold]
    
    if best_matches:
        return inventory_df[inventory_df['product_name'].isin(best_matches)].to_dict('records')
    
    return []

def prepare_stock_response(entities: Entity, user_message: str):
    """
    Docstring for prepare_stock_response
    
    :param entities: The entities extracted from the user's message
    :type entities: Entity
    :param user_message: The original user message
    :type user_message: str
    This function checks the inventory for the given product name
      and returns the available quantity.
    """


    # Use the cleaned data from the pydantic model to search the inventory.
    product_to_find = entities.product_name.lower().strip()

    if product_to_find.endswith('s') and not product_to_find.endswith('ss'):
        singular_version = product_to_find[:-1]
        # We'll keep both, the matching function will handle it
        product_to_find = f"{product_to_find}|{singular_version}"
        print(f"DEBUG: Searching for both plural and singular versions: '{product_to_find}'")

    # Get the category hint
    category = entities.category_hint

    # Get thhe actual catogory from the hint using the function we defined in stock_check.py
    category = extract_category_from_query(category)

    # Get the data from the get inventory_data function to use in the system prompt for the LLM.
    results =get_proactive_matches(product_to_find, inventory_df, threshold=65)

    # Format the results into sting for the llm 
    if not results:
        context = f"No matching products found in inventory for search term: {product_to_find}"
    else:
        context = "Found these items:\n" + "\n".join([str(r) for r in results])

    # Set up the system prompt for the LLM with the context and instructions for intent detection.
    inventory_prompt = """
    You are a helpful grocery store assistant. 

    The users original message is: "{user_message}"

    The user searched for: "{product_to_find}"

    Here are the matching products from our inventory:
    {context}

    YOUR TASK:
    1. Select products that match what exactly what the user asked for.
    2. When in doubt, INCLUDE the product - let the user decide
    3. Only exclude products that are clearly unrelated (e.g., "apple juice" when searching for "apple sauce")
    4. Include all variants and related products (e.g., whole milk, skim milk, chocolate milk, cashew milk all count as "milk")
    5. Provide a friendly, natural response. ALWAYS list the products you found, their prices, and quantities using PLAIN TEXT with newlines, NOT HTML or Markdown.
    6. If no products are found, respond with a friendly message saying you couldn't find any matches, and maybe suggest checking the spelling or trying a different search term.
    7. If you have a variety of matches, introduce them and then list them out with each product on a new line. For example:
       "We have several milk options available! Here's what we found:
       - Whole milk ($350)
       - Skim milk ($300)
       - Chocolate milk ($400)
       - Cashew milk ($450)"
    8. At the end of your response, ask the user if they would like to add any of the found products to their cart.
    9. IMPORTANT: Use only plain text formatting. Do NOT use HTML tags like <br>, <b>, or any other HTML elements. Use newlines and plain text for formatting.

    OUTPUT FORMAT (JSON only, no other text):
    {{
        "message": "A friendly message to show the user",
        "products": [
            {{
                "product_name": "exact name from inventory",
                "quantity": available quantity,
                "price": price
            }}
        ],
        "action_ready": true/false  // true if these products can be added to cart directly
    }}

    EXAMPLES:
    
    User searched: "apple sauce"
    Inventory: [Apple Sauce, Unsweetened Applesauce, Apple Juice]
    {{
        "message": "We have a few apple sauce options in stock:\\n- Apple Sauce ($150)\\n- Unsweetened Applesauce ($120)\\n\\nWould you like to add either of these to your cart?",
        "products": [
            {{"product_name": "Apple Sauce", "quantity": 25, "price": 150}},
            {{"product_name": "Unsweetened Applesauce", "quantity": 10, "price": 120}}
        ],
        "action_ready": true
    }}
    
    User searched: "ackee"
    Inventory: [Ackee - Canned, Ackee - Frozen, Blackberries - Fresh]
    {{
        "message": "We have ackee available! We've got:\\n- Canned Ackee ($450)\\n- Frozen Ackee ($550)\\n\\nWould you like to add either of these to your cart?",
        "products": [
            {{"product_name": "Ackee - Canned", "quantity": 35, "price": 450}},
            {{"product_name": "Ackee - Frozen", "quantity": 20, "price": 550}}
        ],
        "action_ready": true
    }}
    
    Now process the user's search.
    """.format(user_message=user_message, product_to_find=product_to_find, context=context)
    
    return inventory_prompt


if __name__ == "__main__":
    
    # Debug: First, let's see what's ACTUALLY in the CSV for oxtail
    print("=" * 60)
    print("DEBUG: Checking what's in the CSV for 'oxtail'")
    print("=" * 60)
    
    # Search the CSV directly
    oxtail_in_csv = inventory_df[inventory_df['product_name'].str.lower().str.contains('oxtail', na=False)]
    if not oxtail_in_csv.empty:
        print(f"Found in CSV: {oxtail_in_csv[['product_name', 'quantity', 'price']].to_dict('records')}")
    else:
        print("WARNING: Nothing with 'oxtail' found in CSV at all!")
        print("Here are the first 10 product names in your CSV:")
        print(inventory_df['product_name'].head(10).tolist())
    
    print("\n" + "=" * 60)
    print("TESTING get_proactive_matches() FUNCTION")
    print("=" * 60)
    
    test_cases = [
        {"msg": "do you have oxtails", "prod": "oxtails", "cat": "meat"},
        {"msg": "i need oxtail", "prod": "oxtail", "cat": "meat"},
        {"msg": "ox tail", "prod": "ox tail", "cat": "meat"},
        {"msg": "do you have any pork?", "prod": "pork", "cat": "meat"},
    ]
    
    for case in test_cases:
        print(f"\nTesting: '{case['msg']}'")
        print(f"  Search term: '{case['prod']}'")
        
        # Use YOUR new function, NOT the old one
        results = get_proactive_matches(case['prod'], inventory_df, threshold=65)
        
        print(f"  Results found: {len(results)}")
        if results:
            for r in results:
                print(f"    → {r['product_name']} (qty: {r['quantity']}, price: ${r['price']})")
        else:
            print(f"  ⚠️ No matches found for '{case['prod']}'")
            
            # Debug: Show stemmed version
            stemmed = stemmer.stem(case['prod'].lower())
            print(f"  Debug: Stemmed version = '{stemmed}'")
            
            # Debug: Show what stems exist in CSV
            all_stems = set()
            for prod in inventory_df['product_name'].tolist():
                all_stems.add(stemmer.stem(prod.lower()))
            print(f"  Debug: Available stems in CSV: {sorted([s for s in all_stems if 'oxtail' in s or 'tail' in s])}")
    