from datasets.data import inventory_df, sales_df
from services.context_handler import get_user_state
from typing import Optional, List, Dict, Any
from rapidfuzz import process, fuzz

from difflib import SequenceMatcher
import re
import random
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()



def normalize(name: str) -> str:
    # Remove any special characters
    name = re.sub(r'[^\w\s]', '', name)
    return name.lower().replace("-", " ").strip()


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


def rank_results(results: list, top_n: int = 3, boost_n: int = 2) -> list:
    """
    Merges sales data into results, ranks by units_sold.
    Returns top_n best sellers + boost_n slow movers (deduplicated).
    """
    # Join sales data onto results by product_code
    sales_lookup = sales_df.set_index("product_code")["units_sold"].to_dict()

    for r in results:
        r["units_sold"] = sales_lookup.get(r["product_code"], 0)

    sorted_results = sorted(results, key=lambda x: x["units_sold"], reverse=True)

    top_sellers = sorted_results[:top_n]
    slow_movers = sorted_results[-boost_n:]

    # Deduplicate
    seen    = {r["product_code"] for r in top_sellers}
    boosted = [r for r in slow_movers if r["product_code"] not in seen]

    final = top_sellers + boosted

    # Clean up units_sold before returning — frontend doesn't need it
    for r in final:
        r.pop("units_sold", None)

    return final


def get_inventory_data(search_term, category_hint: str = None) -> list:
    if not search_term or len(search_term) < 2:
        return []

    search_lower   = search_term.lower().strip()
    inventory_names = inventory_df["product_name"].tolist()

    # Pass 1: exact word / prefix match
    exact_matches = [
        name for name in inventory_names
        if search_lower in normalize(name).split()
        or normalize(name).startswith(search_lower)
    ]

    # Pass 2: fuzzy fallback
    if not exact_matches:
        normalized_inventory = [normalize(n) for n in inventory_names]
        raw = process.extract(normalize(search_term), normalized_inventory, scorer=fuzz.WRatio)
        matched_indices = [
            idx for _, score, idx in raw
            if score >= 88 and normalize(search_term) in normalize(inventory_names[idx])
        ]
        exact_matches = [inventory_names[i] for i in matched_indices]

    if not exact_matches:
        return []

    matches_df = inventory_df[inventory_df["product_name"].isin(exact_matches)]

    # Filter by category if provided
    if category_hint:
        cat_filtered = matches_df[matches_df["category"].str.lower() == category_hint.lower()]
        if not cat_filtered.empty:
            matches_df = cat_filtered

    results = matches_df[["product_code", "product_name", "quantity", "price"]].to_dict(orient="records")

    # Rank: top sellers + slow movers boost
    return rank_results(results)


def get_product_aisle(product_name: str) -> Optional[Dict[str, Any]]:
    """
    Get aisle information for a product.
    User asks: "Where can I find bread?" or "What aisle is milk in?"
    
    Returns: {
        "product_name": "Bread",
        "aisle": 5,
        "section": "Bakery",
        "location_details": "Left side, middle shelf"
    }
    """
    if not product_name or len(product_name) < 2:
        return None
    
    search_lower = product_name.lower().strip()
    inventory_names = inventory_df["product_name"].tolist()
    
    # Pass 1: exact word / prefix match
    exact_matches = [
        name for name in inventory_names
        if search_lower in normalize(name).split()
        or normalize(name).startswith(search_lower)
    ]
    
    # Pass 2: fuzzy fallback
    if not exact_matches:
        normalized_inventory = [normalize(n) for n in inventory_names]
        raw = process.extract(normalize(product_name), normalized_inventory, scorer=fuzz.WRatio)
        matched_indices = [
            idx for _, score, idx in raw
            if score >= 88 and normalize(product_name) in normalize(inventory_names[idx])
        ]
        exact_matches = [inventory_names[i] for i in matched_indices]
    
    if not exact_matches:
        return None
    
    # Get the first match
    matched_product = exact_matches[0]
    product_row = inventory_df[inventory_df["product_name"] == matched_product]
    
    if product_row.empty:
        return None
    
    row = product_row.iloc[0]
    
    # Extract aisle from location column if it exists, otherwise use category
    aisle = None
    location_details = ""
    
    if "aisle" in inventory_df.columns:
        aisle = row.get("aisle")
    
    if "location" in inventory_df.columns:
        location_details = row.get("location", "")
    
    # If no aisle column, map category to typical aisle
    if not aisle:
        aisle = map_category_to_aisle(row.get("category", ""))
    
    return {
        "product_name": row["product_name"],
        "category": row.get("category", ""),
        "aisle": aisle,
        "location_details": location_details or get_default_location(row.get("category", ""))
    }


def map_category_to_aisle(category: str) -> Optional[int]:
    """
    Map product category to typical supermarket aisle number.
    Customize based on your store layout.
    """
    category_aisle_map = {
        "Produce": 1,
        "Meat & Poultry": 2,
        "Seafood": 3,
        "Dairy": 4,
        "Bakery": 5,
        "Grains & Rice": 6,
        "Pasta & Noodles": 6,
        "Canned Goods": 7,
        "Frozen": 8,
        "Beverages": 9,
        "Snacks": 10,
        "Condiments": 11,
        "Spices & Seasonings": 11,
        "Baking": 12,
        "International Foods": 13,
        "Health & Beauty": 14,
        "Household": 15,
    }
    
    return category_aisle_map.get(category)


def get_default_location(category: str) -> str:
    """Get default location description based on category"""
    location_map = {
        "Produce": "Front of store",
        "Meat & Poultry": "Back left section",
        "Seafood": "Back right section",
        "Dairy": "Back wall",
        "Bakery": "Right side",
        "Beverages": "Back, near checkout",
        "Frozen": "Back wall, bottom shelves",
        "Snacks": "Center aisles",
    }
    
    return location_map.get(category, "Check store directory")


# Was the follow up a stock check
def was_stock_check(user_message: str, user_id: str) -> Optional[str]:
    """Check if follow-up message is a yes/no response to stock check"""
    
    STANDALONE_YES_NO_RESPONSES = [
        "yes", "yep", "yeah", "yup", "sure", "ok", "okay", "alright", "affirmative",
        "no", "nope", "nah", "negative", "not really", "never",
        "got it", "alrighty", "sounds good", "cool",
        "thanks", "thank you", "roger", "understood", "right", 'hmm', 'hmmm'
    ]

    STANDALONE_YES_NO_RESPONSES_MESSAGES = [
        "I'm not sure which one you mean. Could you tell me which product you want?",
        "Oops! Could you clarify which item you're referring to?",
        "I got your yes/no, but I need to know which product to add. Which one would you like?",
        "Thanks! Can you specify which one from the list you want?",
        "Just to be clear, which product should I go with?",
        "I heard you, but could you confirm the item you mean?",
        "Almost there! Which of the items would you like me to add?",
        "Got it, but I need to know which product you're choosing. Can you specify?",
        "Thanks for your response! Please tell me which product you're referring to.",
        "Okay! Could you clarify which item from the list you want me to add?",
    ]

    # Check if the last intent was stock check
    user_state = get_user_state(user_id)
    last_intent = user_state["last_intent"]
    
    if last_intent == "stock_check":
        # If there is an active pending product selection, do not intercept yes/no here
        if user_state.get("pending_selection"):
            return None

        # Check if the current message is a standalone yes/no response
        for response in STANDALONE_YES_NO_RESPONSES:
            if user_message.lower() == response:
                last_response = user_state.get("last_response", "")
                return random.choice(STANDALONE_YES_NO_RESPONSES_MESSAGES) + "\n\n" + last_response
        
        return None
    
    return None


def normalize(name: str) -> str:
    """Normalize string for comparison"""
    name = re.sub(r'[^\w\s]', '', name)
    return name.lower().replace("-", " ").strip()

from typing import Optional
from nltk.stem import PorterStemmer
from rapidfuzz import fuzz
import re

def extract_subcategory_from_response(user_message: str, available_subcategories: List[str]) -> Optional[str]:
    """
    Extract subcategory from user's response with fuzzy matching, stemming, and ordinal support.
    
    NOW TAKES: available_subcategories list as parameter (from session state)
    
    Args:
        user_message: The user's input (e.g., "oxtail", "first one", "do you have beef")
        available_subcategories: List of valid subcategory names from session
                                 (e.g., ["Chicken", "Beef", "Oxtail", ...])
    
    Returns:
        str: The matched subcategory name, or None to trigger LLM fallback
    
    Examples:
    - extract_subcategory_from_response("oxtail", ["Chicken", "Beef", "Oxtail"]) → "Oxtail"
    - extract_subcategory_from_response("first one", ["Chicken", "Beef"]) → "Chicken"
    - extract_subcategory_from_response("ostails", ["Chicken", "Beef", "Oxtail"]) → "Oxtail"
    - extract_subcategory_from_response("xyz", ["Chicken", "Beef"]) → None (LLM fallback)
    """
    
    if not available_subcategories:
        print("[WARNING] extract_subcategory_from_response: No subcategories provided")
        return None
    
    stemmer = PorterStemmer()
    user_lower = user_message.lower().strip()
    
    # ========== ORDINAL MATCHING (handles "first one", "1st", "number 1", etc.) ==========
    ordinal_matches = {
        r'\b(first|1st|number\s*1|one)\b': 0,
        r'\b(second|2nd|number\s*2|two)\b': 1,
        r'\b(third|3rd|number\s*3|three)\b': 2,
        r'\b(fourth|4th|number\s*4|four)\b': 3,
        r'\b(fifth|5th|number\s*5|five)\b': 4,
        r'\b(sixth|6th|number\s*6|six)\b': 5,
        r'\b(seventh|7th|number\s*7|seven)\b': 6,
        r'\b(eighth|8th|number\s*8|eight)\b': 7,
        r'\b(ninth|9th|number\s*9|nine)\b': 8,
        r'\b(tenth|10th|number\s*10|ten)\b': 9,
    }
    
    for pattern, index in ordinal_matches.items():
        if re.search(pattern, user_lower):
            if 0 <= index < len(available_subcategories):
                matched = available_subcategories[index]
                print(f"[SUBCATEGORY] Ordinal match: '{user_message}' → {matched}")
                return matched
    
    # ========== EXACT SUBSTRING MATCH (best case) ==========
    for subcategory in available_subcategories:
        if subcategory.lower() in user_lower or user_lower in subcategory.lower():
            print(f"[SUBCATEGORY] Exact match: '{user_message}' → {subcategory}")
            return subcategory
    
    # ========== STEM-BASED MATCHING ==========
    def stem_normalize(word: str) -> str:
        """Normalize word using Porter stemmer"""
        return stemmer.stem(word.lower())
    
    def extract_keywords(text: str) -> list:
        """Extract meaningful keywords from text, excluding stop words"""
        EXCLUDED_WORDS = {
            "the", "a", "an", "some", "any", "you", "have", "please", "show",
            "want", "stock", "inventory", "check", "got", "gotten", "available",
            "our", "what", "which", "do", "you", "have", "is", "i", "me", "my",
            "di", "fuss", "one", "first", "second", "third", "1st", "2nd", "3rd",
            "number", "di fuss"
        }
        
        words = text.split()
        IMPORTANT_SHORT = {"ham", "hot", "pig", "nut", "egg"}
        
        keywords = [
            w for w in words
            if (len(w) > 2 or w.lower() in IMPORTANT_SHORT) and w.lower() not in EXCLUDED_WORDS
        ]
        return keywords
    
    normalized_message_stems = [stem_normalize(kw) for kw in extract_keywords(user_message)]
    
    for subcategory in available_subcategories:
        normalized_subcategory = stem_normalize(subcategory)
        
        # Check if any keyword stem is in the subcategory stem
        for msg_stem in normalized_message_stems:
            if msg_stem in normalized_subcategory or normalized_subcategory in msg_stem:
                print(f"[SUBCATEGORY] Stem match: '{user_message}' → {subcategory}")
                return subcategory
    
    # ========== FUZZY MATCHING ==========
    best_match = None
    best_score = 0
    
    for subcategory in available_subcategories:
        fuzzy_score = fuzz.ratio(user_lower, subcategory.lower())
        partial_score = fuzz.partial_ratio(user_lower, subcategory.lower())
        
        # Combine scores (partial gets 40% weight for typo tolerance)
        combined_score = (fuzzy_score * 0.6) + (partial_score * 0.4)
        
        if combined_score > best_score and combined_score >= 70:  # 70% threshold
            best_score = combined_score
            best_match = subcategory
    
    if best_match:
        print(f"[SUBCATEGORY] Fuzzy match: '{user_message}' → {best_match} (score: {best_score:.1f})")
        return best_match
    
    # ========== NO MATCH FOUND ==========
    print(f"[SUBCATEGORY] No match found for '{user_message}'. Best score was {best_score:.1f}. Falling back to LLM...")
    return None  # Caller will handle LLM fallback


# ========== HELPER: BUILD DYNAMIC SUBCATEGORY MAP FROM INVENTORY ==========
def build_category_subcategory_map(inventory_df) -> dict:
    """
    Build a dynamic mapping from inventory data:
    {
        "Meat & Poultry": ["Chicken", "Beef", "Oxtail", ...],
        "Produce": ["Apple", "Banana", "Carrot", ...],
        ...
    }
    
    Args:
        inventory_df: Pandas DataFrame with columns: category, product_name, etc.
    
    Returns:
        dict: {category: [subcategories]}
    """
    category_map = {}
    
    for _, row in inventory_df.iterrows():
        category = row.get("category")
        product_name = row.get("product_name")
        
        if not category or not product_name:
            continue
        
        # Extract subcategory from product name
        # Strategy: Use first word as default fallback
        words = product_name.split()
        subcategory = words[0] if words else None
        
        # Optional: Keyword-based extraction for better accuracy
        product_lower = product_name.lower()
        keyword_map = {
            "chicken": "Chicken", "beef": "Beef", "pork": "Pork",
            "oxtail": "Oxtail", "fish": "Fish", "shrimp": "Shrimp",
            "milk": "Milk", "cheese": "Cheese", "bread": "Bread",
            "rice": "Rice", "juice": "Juice", "water": "Water",
            "turkey": "Turkey", "lamb": "Lamb", "duck": "Duck",
            "apple": "Apple", "banana": "Banana", "carrot": "Carrot",
            "tomato": "Tomato", "lettuce": "Lettuce", "broccoli": "Broccoli",
        }
        
        for keyword, mapped_name in keyword_map.items():
            if keyword in product_lower:
                subcategory = mapped_name
                break
        
        if not subcategory:
            continue
        
        # Add to map
        if category not in category_map:
            category_map[category] = set()
        category_map[category].add(subcategory)
    
    # Convert sets → sorted lists
    return {
        cat: sorted(list(subs))
        for cat, subs in category_map.items()
    }



def extract_category_from_query(user_message: str) -> str:
    """Extract main category from user's question"""
    user_lower = user_message.lower()

    from nltk.stem import PorterStemmer
    from rapidfuzz import fuzz
    stemmer = PorterStemmer()

    def stem_normalize(word):
        return stemmer.stem(word)

        
    category_keywords = {
    # Meat & Seafood
    "meat": "Meat & Poultry",
    "poultry": "Meat & Poultry",
    "chicken": "Meat & Poultry",
    "beef": "Meat & Poultry",
    "pork": "Meat & Poultry",
    "seafood": "Seafood",
    "fish": "Seafood",
    "shrimp": "Seafood",
    "salmon": "Seafood",

    # Produce & Fruits
    "produce": "Produce",
    "vegetable": "Produce",
    "veggie": "Produce",
    "fruit": "Fruits",
    "apple": "Fruits",
    "banana": "Fruits",

    # Dairy & Eggs
    "dairy": "Dairy & Eggs",
    "milk": "Dairy & Eggs",
    "cheese": "Dairy & Eggs",
    "egg": "Dairy & Eggs",
    "butter": "Dairy & Eggs",

    # Bakery & Breakfast
    "bakery": "Bakery",
    "bread": "Bakery",
    "bun": "Bakery",
    "cake": "Bakery",
    "breakfast": "Breakfast",
    "cereal": "Breakfast",
    "oats": "Breakfast",

    # Grains & Dry Goods
    "grain": "Grains & Rice",
    "rice": "Grains & Rice",
    "flour": "Baking Supplies",
    "pasta": "Dry Goods",
    "macaroni": "Dry Goods",

    # Pantry / Cooking
    "spice": "Spices & Seasonings",
    "seasoning": "Spices & Seasonings",
    "salt": "Spices & Seasonings",
    "sugar": "Sugar & Sweeteners",
    "sweetener": "Sugar & Sweeteners",
    "syrup": "Sugar & Sweeteners",
    "baking": "Baking Supplies",
    "yeast": "Baking Supplies",

    # Canned / Jarred / Condiments
    "canned": "Canned & Jarred",
    "tin": "Canned & Jarred",
    "sauce": "Condiments & Sauces",
    "ketchup": "Condiments & Sauces",
    "mayo": "Condiments & Sauces",
    "mustard": "Condiments & Sauces",
    "vinegar": "Vinegar",

    # Snacks & Drinks
    "snack": "Snacks",
    "chips": "Snacks",
    "biscuit": "Snacks",
    "beverage": "Beverages",
    "juice": "Beverages",
    "soda": "Beverages",
    "water": "Beverages",

    # Frozen / Prepared
    "frozen": "Frozen Foods",
    "ice cream": "Frozen Foods",
    "prepared": "Prepared Foods",
    "ready meal": "Prepared Foods",
    "meal kit": "Meal Kits",

    # Specialty / Health / International
    "organic": "Health Foods",
    "health": "Health Foods",
    "international": "International Foods",
    "imported": "International Foods",
    "specialty": "Specialty",

    # Nuts / Seeds
    "nut": "Nuts & Seeds",
    "seed": "Nuts & Seeds",
    "peanut": "Nuts & Seeds",

    # Household / Other
    "cleaning": "Household",
    "detergent": "Household",
    "household": "Household",
    "pet": "Pet Supplies",
    "dog": "Pet Supplies",
    "cat": "Pet Supplies",
    "tobacco": "Tobacco",}
 
    
    def categorize_word(word):
        words = word.split()

        # Excluded words to speed up search
        EXCLUDED_WORDS = {"the", "some", "any", "you", "have", "please", "show","want", 'stock', 'inventory', 'check', 'got', 'gotten', 'available', 'our', 'what', 'which'}
        IMPORTANT_SHORT_WORDS = {"egg", "milk", "nut", "seed", "tea", "ham"}

        # Remove words less than <=2 characters (except for important ones like "egg", "milk")
        norm_words = [stem_normalize(w)
            for w in words
            if (
                (len(w) > 2 or w.lower() in IMPORTANT_SHORT_WORDS)  # allow longer OR important short
                and w.lower() not in EXCLUDED_WORDS                 # exclude stop words
            )]

        for keyword, cat in category_keywords.items():
            norm_keyword =  stem_normalize(keyword)
            print(f"User Words for category extraction: {norm_words}, Checking against keyword: '{keyword}' (normalized: '{norm_keyword}')")

            for nw in norm_words:
                if norm_keyword in nw or nw in norm_keyword:
                    if fuzz.ratio(word, norm_keyword) >= 80:
                        return cat, True
    
        return None, False

    # Call the categorize_word function on the entire user message and also on individual words to find the best match
    category, success = categorize_word(user_message)
    if not success or category is None:
        return "", False  # Return empty string (falsy) so calling code skips category logic
    return category, True



def get_subcategories_in_category(category: str) -> List[Dict[str, Any]]:
    """
    Get unique subcategories in a category.
    For Grains & Rice: extract "Jasmine", "Basmati", "Brown", etc. (not just "Rice")
    """
    if not category:
        return []
    
    category_products = inventory_df[
        inventory_df['category'].str.lower() == category.lower()
    ]
    
    if category_products.empty:
        return []
    
    subcategories = {}
    
    for _, row in category_products.iterrows():
        product_name = row['product_name']
        words = product_name.split()
        
        # IMPORTANT: For products like "Jasmine Rice", extract "Jasmine" not "Rice"
        # For "Boneless Skinless Chicken Breast", extract "Chicken"
        
        main_type = None
        
        # Check for specific keywords FIRST
        type_keywords = {
            "chicken": "Chicken", "beef": "Beef", "pork": "Pork",
            "salmon": "Salmon", "fish": "Fish", "shrimp": "Shrimp",
            "ham": "Ham", "sausage": "Sausage", "turkey": "Turkey",
            "duck": "Duck", "lamb": "Lamb", "bison": "Bison",
            "jasmine": "Jasmine", "basmati": "Basmati", "brown": "Brown",
            "arborio": "Arborio", "sushi": "Sushi", "thai": "Thai", "wild": "Wild",
            "long": "Long Grain", "short": "Short Grain", "white": "White",
        }
        
        product_lower = product_name.lower()
        for keyword, type_name in type_keywords.items():
            if keyword in product_lower:
                main_type = type_name
                break
        
        # If no keyword match, use first word (fallback)
        if not main_type:
            main_type = words[0] if words else product_name
        
        # Count products
        if main_type not in subcategories:
            subcategories[main_type] = 0
        subcategories[main_type] += 1
    
    # Sort by count
    result = [
        {"subcategory": name, "count": count}
        for name, count in subcategories.items()
    ]
    result.sort(key=lambda x: x['count'], reverse=True)
    
    return result

def get_products_by_subcategory(category: str, subcategory: str) -> List[Dict[str, Any]]:
    """
    Get all products in a category AND subcategory.
    Example: category="Meat & Poultry", subcategory="Chicken" → all chicken products
    """
    if not category or not subcategory:
        return []
    
    category_products = inventory_df[
        inventory_df['category'].str.lower() == category.lower()
    ]
    
    if category_products.empty:
        return []
    
    # Filter by subcategory (product name contains the subcategory)
    filtered = category_products[
        category_products['product_name'].str.lower().str.contains(subcategory.lower(), na=False)
    ]
    
    results = []
    for _, row in filtered.iterrows():
        results.append({
            "product_name": row['product_name'],
            "price": float(row['price']),
            "quantity": int(row['quantity']),
            "category": row['category'],
            "in_stock": int(row['quantity']) > 0
        })
    
    results.sort(key=lambda x: (-x['in_stock'], x['product_name']))
    
    return results


def clean_stock_query(text: str) -> str:
    text = text.lower().strip()
    
    # Remove filler words
    remove_words = [
        "the", "some", "any", "a", "an",
        "show", "me", "give", "i", "want",
        "do", "you", "have", "please"
    ]
    
    tokens = text.split()
    tokens = [t for t in tokens if t not in remove_words]
    
    cleaned = " ".join(tokens)
    
    # Normalize plurals (VERY IMPORTANT)
    if cleaned.endswith("s"):
        cleaned = cleaned[:-1]
    
    return cleaned.strip()


def normalize_selection_text(text: str) -> str:
    """
    Cleans user selection input for better matching.
    Example:
    'the oxtails' -> 'oxtail'
    'some beef' -> 'beef'
    """
    text = text.lower().strip()

    # Remove filler words
    stop_words = {
        "the", "a", "an", "some", "any", "please",
        "i", "want", "need", "get", "me", "give"
    }

    tokens = [w for w in re.findall(r"\w+", text) if w not in stop_words]

    # Join back
    cleaned = " ".join(tokens)

    # Singularize simple plurals (VERY IMPORTANT)
    if cleaned.endswith("s"):
        cleaned = cleaned[:-1]

    return cleaned.strip()

def is_subcategory_selection(user_message: str, user_id: str) -> bool:
    """Detect subcategory selection with better matching"""
    state = get_user_state(user_id)

    if state.get("last_intent") not in ["inventory_check", "stock_check", "general_chat", "add_to_cart", "follow_up"]:
        return False

    last_subcategories = state.get("last_subcategories", [])
    if not last_subcategories:
        return False

    text = normalize_selection_text(user_message).lower()

    # Get all subcategory names for matching
    sub_names = [s['subcategory'].lower() for s in last_subcategories]

    # EXACT MATCH (most important)
    if text in sub_names:
        return True

    # Substring match
    for sub_name in sub_names:
        if text in sub_name or sub_name in text:
            return True

    # Fuzzy match
    for sub_name in sub_names:
        similarity = SequenceMatcher(None, text, sub_name).ratio()
        if similarity >= 0.75:  # Increased threshold
            return True

    # Ordinal selection
    if any(o in user_message.lower() for o in ["first", "1st", "second", "2nd", "third", "3rd"]):
        return True

    return False


def get_selected_subcategory(user_message: str, user_id: str) -> Optional[str]:
    """Get selected subcategory with exact matching"""
    state = get_user_state(user_id)
    last_subcategories = state.get("last_subcategories", [])

    text = normalize_selection_text(user_message).lower()

    # EXACT MATCH FIRST
    for sub in last_subcategories:
        if text == sub['subcategory'].lower():
            return sub['subcategory']

    # SUBSTRING
    for sub in last_subcategories:
        if text in sub['subcategory'].lower() or sub['subcategory'].lower() in text:
            return sub['subcategory']

    # FUZZY
    for sub in last_subcategories:
        similarity = SequenceMatcher(None, text, sub['subcategory'].lower()).ratio()
        if similarity >= 0.75:
            return sub['subcategory']

    # ORDINAL
    if "first" in text and len(last_subcategories) >= 1:
        return last_subcategories[0]['subcategory']
    if "second" in text and len(last_subcategories) >= 2:
        return last_subcategories[1]['subcategory']

    return None

def is_product_selection(user_message: str, user_id: str) -> bool:
    """
    Detect if user is selecting a specific product (after subcategory drill-down).
    """
    state = get_user_state(user_id)
    
    last_products = state.get("last_inventory_products", [])
    if not last_products:
        return False
    
    text = user_message.lower().strip()
    
    # Check for "all"
    if any(word in text for word in ["all", "everything"]):
        return True
    
    # Fuzzy match with product names
    for product in last_products:
        product_lower = product['product_name'].lower()
        if text in product_lower or product_lower in text:
            return True
        similarity = SequenceMatcher(None, text, product_lower).ratio()
        if similarity >= 0.65:
            return True
    
    # Ordinal
    ordinals = ["first", "1st", "the first", "second", "2nd", "the second"]
    if any(o in text for o in ordinals):
        return True
    
    return False


def get_selected_products(user_message: str, user_id: str) -> List[str]:
    """Get product names user selected"""
    state = get_user_state(user_id)
    last_products = state.get("last_inventory_products", [])
    text = user_message.lower().strip()
    selected = []
    
    # Check "all"
    if any(word in text for word in ["all", "everything"]):
        return [p['product_name'] for p in last_products]
    
    # Fuzzy match
    for product in last_products:
        product_lower = product['product_name'].lower()
        if text in product_lower or product_lower in text:
            selected.append(product['product_name'])
        else:
            similarity = SequenceMatcher(None, text, product_lower).ratio()
            if similarity >= 0.65:
                selected.append(product['product_name'])
    
    # Ordinal
    if not selected:
        if any(o in text for o in ["first", "1st", "the first"]) and len(last_products) >= 1:
            selected.append(last_products[0]['product_name'])
        elif any(o in text for o in ["second", "2nd", "the second"]) and len(last_products) >= 2:
            selected.append(last_products[1]['product_name'])
    
    return selected


async def prepare_inventory_check_response(entities, user_message: str, user_id: str = None) -> str:
    """
    Smart inventory check with two-step drill-down.
    Step 1: Show subcategories (Chicken, Beef, Pork, etc.)
    Step 2: Show detailed products when user picks a subcategory
    """
    state = get_user_state(user_id) if user_id else {}
    
    # STEP 2: User selecting a product from detailed list
    if user_id and is_product_selection(user_message, user_id):
        selected_names = get_selected_products(user_message, user_id)
        
        if selected_names:
            product_info = ""
            for name in selected_names:
                product = next((p for p in state.get("last_inventory_products", []) if p['product_name'] == name), None)
                if product:
                    stock_status = "✅ In Stock" if product['in_stock'] else "❌ Out of Stock"
                    product_info += f"\n• {product['product_name']} — ${product['price']} ({stock_status})"
            
            return f"""
            You are a helpful UCC Supermarket assistant.
            User selected: "{user_message}"
            
            Product details:
            {product_info}
            
            INSTRUCTIONS:
            - Confirm their selection
            - Offer to add to cart, see recipes, or browse more
            - Be conversational
            
            OUTPUT (JSON only):
            {{
                "message": "Great choice! The {selected_names[0]} is ${state['last_inventory_products'][0]['price']}. Would you like to add it to your cart or see recipes using it?",
                "product_names": {selected_names},
                "action_ready": false
            }}
            """
    
    # STEP 1.5: User selecting a subcategory (e.g., "chicken" after "what meats")
    if user_id and is_subcategory_selection(user_message, user_id):
        state = get_user_state(user_id)
        last_category = state.get("last_category")
        selected_subcategory = get_selected_subcategory(user_message, user_id)
        
        if selected_subcategory and last_category:
            # Get products in this subcategory
            products = get_products_by_subcategory(last_category, selected_subcategory)
            
            if products:
                state["last_inventory_products"] = products
                
                product_list = ""
                for p in products:
                    stock = "✅" if p['in_stock'] else "❌"
                    product_list += f"\n• {p['product_name']} — ${p['price']} {stock}"
                
                return f"""
                You are a helpful UCC Supermarket assistant.
                User asked about {selected_subcategory} in {last_category}.
                
                Products:
                {product_list}
                
                INSTRUCTIONS:
                - List the products conversationally (NOT as bullet points)
                - Mention prices
                - Ask which specific product they want or suggest "the first one", "the second one"
                
                OUTPUT (JSON only):
                {{
                    "message": "Great! Here's our {selected_subcategory} selection. We have {len(products)} options ranging from ${min(p['price'] for p in products)} to ${max(p['price'] for p in products)}. Which one interests you?",
                    "subcategory": "{selected_subcategory}",
                    "products_found": {len(products)},
                    "action_ready": false
                }}
                """
    
    # STEP 1: User asking "what meats do you have" → Show subcategories
    category = extract_category_from_query(user_message)
    
    if not category:
        return f"""
        You are a helpful UCC Supermarket assistant.
        User asked: "{user_message}"
        
        INSTRUCTIONS:
        - They're asking about a category but it wasn't clear
        - Ask them to specify (meat, produce, dairy, etc.)
        
        OUTPUT (JSON only):
        {{
            "message": "I'd be happy to help! What category are you interested in? We have Meat & Poultry, Seafood, Produce, Dairy, Bakery, and more. What would you like to browse?",
            "action_ready": false
        }}
        """
    
    subcategories = get_subcategories_in_category(category)
    
    if not subcategories:
        return f"""
        You are a helpful UCC Supermarket assistant.
        No products found in {category}.
        
        OUTPUT (JSON only):
        {{
            "message": "Sorry, we don't currently have items in {category}. Would you like to browse another category?",
            "action_ready": false
        }}
        """
    
    # Store subcategories for next turn
    if user_id:
        state = get_user_state(user_id)
        state["last_subcategories"] = subcategories
        state["last_category"] = category
    
    # Build subcategory list
    subcategory_list = ""
    for i, sub in enumerate(subcategories, 1):
        subcategory_list += f"\n• {sub['subcategory']} ({sub['count']} items)"
    
    return f"""
    You are a helpful UCC Supermarket assistant.
    User asked: "{user_message}"
    
    Available subcategories in {category}:
    {subcategory_list}
    
    INSTRUCTIONS:
    - List the types conversationally
    - Ask which type they want to see
    - Make it feel natural, not robotic
    - Suggest they can say the name or "the first one", "the second one"
    
    OUTPUT (JSON only):
    {{
        "message": "Great question! In our {category} section, we have {len(subcategories)} types: {', '.join([s['subcategory'] for s in subcategories])}. Which would you like to explore?",
        "category": "{category}",
        "subcategories": {len(subcategories)},
        "action_ready": false
    }}
    """


def check_and_update_product_selection(user_id: str, user_message: str) -> tuple[str, bool]:
    """
    Validates if a user message contains a product from the inventory.
    Updates user state if found.
    
    Returns:
        (product_name, True) if found
        ("", False) if not found (Fallback to LLM indicator)
    """
    # 1. Clean the input using your existing utility
    search_term = normalize_selection_text(user_message)
    
    if not search_term or len(search_term) < 2:
        return "", False

    inventory_names = inventory_df["product_name"].tolist()
    match_found = []

    # 2. Pass 1: Exact / Prefix Match (High Performance)
    # Re-using the logic from your get_inventory_data for consistency
    exact_matches = [
        name for name in inventory_names
        if search_term in normalize(name).split()
        or normalize(name).startswith(search_term)
    ]

    if exact_matches:
        match_found = exact_matches[0]

    # 3. Pass 2: Fuzzy Fallback (Typos)
    if not match_found:
        normalized_inventory = [normalize(n) for n in inventory_names]
        # Using RapidFuzz extractOne for better performance/simplicity here
        raw = process.extractOne(
            normalize(search_term), 
            normalized_inventory, 
            scorer=fuzz.WRatio,
            score_cutoff=88 # Using your established threshold
        )
        
        if raw:
            # raw is (string, score, index)
            match_found = inventory_names[raw[2]]

    # 4. State Update and Return
    if match_found:
        state = get_user_state(user_id)
        state["current_product"] = match_found
        
        print(f"Product selection detected: '{user_message}' matched to '{match_found}' in inventory.")
        return match_found, True

    # 5. Fallback indicator
    return "", False